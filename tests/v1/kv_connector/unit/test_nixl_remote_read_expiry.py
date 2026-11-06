# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer must not read remote KV after the producer lease has run out.

Reproduction of https://github.com/vllm-project/vllm/issues/66009 without GPUs.
A router can hold a prefilled request outside the decode engine, so no
heartbeat renews the producer lease. The producer then frees and reuses the
blocks. The stale ``remote_block_ids`` still name that memory, and a NIXL read
would succeed with another request's KV (and, for hybrid models, its recurrent
state). That is cross-request confidentiality and integrity failure, not only
a later NaN or illegal-address crash.

This file is the mocked-clock repro and the regression test:

    .venv/bin/python -m pytest \\
        tests/v1/kv_connector/unit/test_nixl_remote_read_expiry.py -v

On the unfixed tree the consumer posts the read. With the fix it declines,
counts ``vllm:nixl_num_declined_expired_reads``, and marks the request
``nixl_kv_lease_expired`` so the API error tells a router to prefill again.
Heartbeats the consumer itself sent still allow the read.
"""

import time
from unittest.mock import patch

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlConnector,
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    _KV_BLOCKS_EXPIRY_SAFETY_MARGIN,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import HeartbeatInfo
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import kv_lease_extension
from vllm.exceptions import NIXL_KV_LEASE_EXPIRED
from vllm.forward_context import ForwardContext

from .test_nixl_connector import FakeNixlConnectorWorker, FakeNixlWrapper
from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
    make_kv_cache_config,
    make_nixl_scheduler,
)

_REMOTE = FakeNixlConnectorWorker.REMOTE_ENGINE_ID
_REMOTE_REQ = "prefill-req"
_MARGIN = _KV_BLOCKS_EXPIRY_SAFETY_MARGIN
_LEASE = 30
_EXTENSION = kv_lease_extension(_LEASE)

pytestmark = pytest.mark.cpu_test


def _make_connector():
    """Pull worker with a handshaken fake remote and a zero clock offset."""
    block_size = 32
    vllm_config = create_vllm_config(block_size=block_size)
    kv_cache_config = make_kv_cache_config(block_size=block_size, num_blocks=2)
    connector = NixlConnector(vllm_config, KVConnectorRole.WORKER, kv_cache_config)
    connector.connector_worker = FakeNixlConnectorWorker(
        vllm_config,
        connector.engine_id,
        hand_shake_latency=0,
        kv_cache_config=kv_cache_config,
    )
    worker = connector.connector_worker
    assert isinstance(worker.nixl_wrapper, FakeNixlWrapper)
    worker.kv_cache_layout = "LBHNC"
    remote_agents, _ = worker._nixl_handshake(
        host="localhost",
        port=1234,
        remote_tp_size=1,
        expected_engine_id=_REMOTE,
    )
    worker._remote_agents[_REMOTE] = remote_agents
    worker._engine_clock_offset[_REMOTE] = 0.0
    return connector, worker


def _recv_meta(expiry_delta: float | None, local_blocks=([10, 11],)):
    """D-side metadata whose producer deadline is ``expiry_delta`` seconds away."""
    params = {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_block_ids": ([20, 21],),
        "remote_engine_id": _REMOTE,
        "remote_request_id": _REMOTE_REQ,
        "remote_host": "localhost",
        "remote_port": 1234,
        "tp_size": 1,
        "remote_tp_size": 1,
    }
    if expiry_delta is not None:
        params["remote_blocks_expiry_time"] = time.perf_counter() + expiry_delta
        params["remote_lease_extension"] = _EXTENSION
    meta = NixlConnectorMetadata()
    meta.add_new_req_to_recv(
        request_id="req",
        local_block_ids=local_blocks,
        kv_transfer_params=params,
    )
    return meta


def _hb_meta(req_id: str, expiry: float, extension: int = _EXTENSION):
    meta = NixlConnectorMetadata()
    meta.heartbeat_by_engine = {
        _REMOTE: HeartbeatInfo(
            req_ids={req_id},
            host="localhost",
            port=1234,
            tp_size=1,
            leases={req_id: (expiry, extension)},
        )
    }
    return meta


def _load(connector, meta):
    connector.bind_connector_metadata(meta)
    ctx = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
    connector.start_load_kv(ctx)
    return connector.connector_worker.get_transfer_results()


_FAKE_WRAPPER = patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker.NixlWrapper",
    FakeNixlWrapper,
)


@_FAKE_WRAPPER
@pytest.mark.parametrize(
    "expiry_delta, expect_declined",
    [
        (-10.0, True),
        (_MARGIN / 2, True),
        (20.0, False),
        (None, False),
    ],
)
def test_remote_prefill_read_gate(dist_init, expiry_delta, expect_declined):
    """A read whose producer lease has run out must not touch remote memory."""
    connector, worker = _make_connector()
    with patch.object(worker.nixl_wrapper, "make_prepped_xfer") as xfer:
        result = _load(connector, _recv_meta(expiry_delta))
    assert worker.xfer_stats.data["num_declined_expired_reads"] == (
        [1] if expect_declined else []
    )
    assert worker.xfer_stats.data["num_failed_transfers"] == []
    assert worker.xfer_stats.data["num_kv_expired_reqs"] == []
    assert ("req" in result.failed_recving) is expect_declined
    assert ("req" in result.declined_expired_recving) is expect_declined
    if expect_declined:
        xfer.assert_not_called()
        assert "req" in result.finished_recving
    else:
        assert "req" not in result.declined_expired_recving


@_FAKE_WRAPPER
def test_producer_frees_then_consumer_declines(dist_init):
    """Producer reap (blocks reusable) then a consumer read of those ids."""
    producer, producer_worker = _make_connector()
    producer_worker._reqs_to_process.add(_REMOTE_REQ)
    producer_worker._reqs_to_send[_REMOTE_REQ] = time.perf_counter() - 1.0
    freed = producer_worker.get_transfer_results()
    assert _REMOTE_REQ in freed.finished_sending
    assert producer_worker.xfer_stats.data["num_kv_expired_reqs"] == [1]
    del producer

    consumer, consumer_worker = _make_connector()
    with patch.object(consumer_worker.nixl_wrapper, "make_prepped_xfer") as xfer:
        result = _load(consumer, _recv_meta(-10.0))
    xfer.assert_not_called()
    assert result.declined_expired_recving == {"req"}
    assert consumer_worker.xfer_stats.data["num_declined_expired_reads"] == [1]


@_FAKE_WRAPPER
def test_full_local_prefix_hit_skips_gate(dist_init):
    connector, worker = _make_connector()
    result = _load(connector, _recv_meta(-10.0, local_blocks=()))
    assert worker.xfer_stats.data["num_declined_expired_reads"] == []
    assert "req" not in result.failed_recving


@_FAKE_WRAPPER
def test_unknown_clock_offset_reads_as_usual(dist_init):
    connector, worker = _make_connector()
    worker._engine_clock_offset.pop(_REMOTE)
    result = _load(connector, _recv_meta(-10.0))
    assert "req" not in result.declined_expired_recving


@_FAKE_WRAPPER
def test_heartbeat_keeps_an_otherwise_expired_lease(dist_init, monkeypatch):
    """Heartbeats sent inside the lease renew it; a gap that misses it does not."""
    connector, worker = _make_connector()
    now = {"t": 0.0}
    monkeypatch.setattr(time, "perf_counter", lambda: now["t"])
    # Prefill finished at t=0 on a peer with the same clock: deadline t=30.
    meta = _recv_meta(float(_LEASE))
    leases_meta = _hb_meta(_REMOTE_REQ, float(_LEASE))
    for t in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0):
        now["t"] = t
        worker._send_heartbeats(leases_meta)
    # Last renewal is t=25 -> deadline 45. t=35 is outside the 5s margin.
    now["t"] = 35.0
    result = _load(connector, meta)
    assert "req" not in result.declined_expired_recving

    # A heartbeat after the lease lapsed does not revive it.
    lapsed, lapsed_worker = _make_connector()
    now["t"] = 0.0
    lapsed_meta = _recv_meta(float(_LEASE))
    lapsed_worker._send_heartbeats(leases_meta)
    now["t"] = 40.0
    lapsed_worker._send_heartbeats(leases_meta)
    result = _load(lapsed, lapsed_meta)
    assert result.declined_expired_recving == {"req"}


def test_decode_scheduler_ships_lease_with_heartbeats():
    scheduler = make_nixl_scheduler(heartbeat=True)
    req = create_request(request_id=1, do_remote_prefill=True)
    req.kv_transfer_params["remote_blocks_expiry_time"] = 1234.5
    req.kv_transfer_params["remote_lease_extension"] = 20
    scheduler.on_new_request(req)
    engine_id = req.kv_transfer_params["remote_engine_id"]
    info = scheduler._heartbeat_by_engine[engine_id]
    assert info.leases == {"prefill-1": (1234.5, 20)}
    scheduler._stop_heartbeat(req.request_id)
    assert engine_id not in scheduler._heartbeat_by_engine


@pytest.mark.parametrize("kv_lease_duration", [30, 12])
def test_prefill_request_finished_exports_lease_deadline(kv_lease_duration):
    """P exports the lease deadline and how much each heartbeat renews it."""
    vllm_config = create_vllm_config(
        kv_role="kv_producer",
        kv_connector_extra_config={"kv_lease_duration": kv_lease_duration},
    )
    scheduler = create_scheduler(vllm_config)
    block_size = vllm_config.cache_config.block_size
    request = create_request(
        request_id=7,
        num_tokens=block_size * 2,
        block_size=block_size,
        do_remote_decode=True,
    )
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    before = time.perf_counter()
    engine_core_outputs = scheduler.update_from_output(
        scheduler_output, create_model_runner_output([request])
    )
    after = time.perf_counter()
    kv = engine_core_outputs[0].outputs[0].kv_transfer_params
    assert kv is not None and kv["do_remote_prefill"] is True
    expiry = kv["remote_blocks_expiry_time"]
    assert isinstance(expiry, float)
    assert before + kv_lease_duration <= expiry <= after + kv_lease_duration
    assert kv["remote_lease_extension"] == kv_lease_extension(kv_lease_duration)


def test_failed_request_stop_reason_is_the_lease_token():
    """A declined read finishes with stop_reason nixl_kv_lease_expired."""
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import RequestStatus

    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    request = create_request(
        request_id=1,
        num_tokens=32,
        block_size=16,
        do_remote_prefill=True,
    )
    scheduler.add_request(request)
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    model_output = create_model_runner_output(reqs=[])
    model_output.kv_connector_output = KVConnectorOutput(
        finished_recving={request.request_id},
        failed_recving={request.request_id},
        declined_expired_recving={request.request_id},
    )
    outputs = scheduler.update_from_output(SchedulerOutput.make_empty(), model_output)
    finished = outputs[request.client_index].outputs
    assert len(finished) == 1
    assert finished[0].stop_reason == NIXL_KV_LEASE_EXPIRED
    assert request.stop_reason == NIXL_KV_LEASE_EXPIRED
