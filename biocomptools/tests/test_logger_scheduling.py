# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Tests for Logger scheduling methods."""

from biocomptools.toollib.loggers.logger import Logger


def test_should_fire_interval():
    lg = Logger(call_at_interval=10)
    assert not lg.should_fire(0)
    assert not lg.should_fire(5)
    assert lg.should_fire(10)
    assert lg.should_fire(20)
    assert not lg.should_fire(15)


def test_should_fire_call_at():
    lg = Logger(call_at=[5, 50])
    assert not lg.should_fire(0)
    assert lg.should_fire(5)
    assert lg.should_fire(50)
    assert not lg.should_fire(10)


def test_should_fire_union():
    lg = Logger(call_at_interval=10, call_at=[5, -1])
    assert lg.should_fire(5)
    assert lg.should_fire(10)
    assert lg.should_fire(20)
    assert not lg.should_fire(7)


def test_should_fire_no_periodic():
    lg = Logger(call_at_interval=None, call_at=[100])
    assert not lg.should_fire(50)
    assert lg.should_fire(100)


def test_should_fire_start():
    lg = Logger(call_at=[0, -1])
    assert lg.should_fire_start()
    assert lg.should_fire_end()


def test_should_fire_start_default():
    lg = Logger()  # default call_at=[-1]
    assert not lg.should_fire_start()
    assert lg.should_fire_end()


def test_should_fire_end_no():
    lg = Logger(call_at=[0, 100])
    assert lg.should_fire_start()
    assert not lg.should_fire_end()


def test_execution_mode_default():
    lg = Logger()
    assert lg.execution_mode == "thread"


def test_execution_mode_inline():
    lg = Logger(execution_mode="inline")
    assert lg.execution_mode == "inline"


def test_execution_mode_process():
    lg = Logger(execution_mode="process")
    assert lg.execution_mode == "process"
