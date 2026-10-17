import logging
import os
import time
from optparse import Values
from pathlib import Path
from typing import Any, Callable, Iterator, List, NoReturn, Optional
from unittest.mock import Mock, patch

import pytest
from pip._vendor.packaging.version import Version

from pip._internal import self_outdated_check
from pip._internal.cli import cmdoptions
from pip._internal.cli.base_command import Command
from pip._internal.cli.req_command import IndexGroupCommand
from pip._internal.cli.status_codes import SUCCESS
from pip._internal.utils import temp_dir
from pip._internal.utils.logging import BrokenStdoutLoggingError
from pip._internal.utils.temp_dir import TempDirectory


@pytest.fixture
def fixed_time() -> Iterator[None]:
    with patch("time.time", lambda: 1547704837.040001 + time.timezone):
        yield


class FakeCommand(Command):
    _name = "fake"

    def __init__(
        self, run_func: Optional[Callable[[], int]] = None, error: bool = False
    ) -> None:
        if error:

            def run_func() -> int:
                raise SystemExit(1)

        self.run_func = run_func
        super().__init__(self._name, self._name)

    def main(self, args: List[str]) -> int:
        args.append("--disable-pip-version-check")
        return super().main(args)

    def run(self, options: Values, args: List[str]) -> int:
        logging.getLogger("pip.tests").info("fake")
        # Return SUCCESS from run if run_func is not provided
        if self.run_func:
            return self.run_func()
        else:
            return SUCCESS


class FakeCommandWithUnicode(FakeCommand):
    _name = "fake_unicode"

    def run(self, options: Values, args: List[str]) -> int:
        logging.getLogger("pip.tests").info(b"bytes here \xE9")
        logging.getLogger("pip.tests").info(b"unicode here \xC3\xA9".decode("utf-8"))
        return SUCCESS


class FakeIndexGroupCommand(IndexGroupCommand):
    """A command that does the pip self version check, like ``pip install``."""

    _name = "fake_index_group"

    def __init__(self, run_func: Callable[[], int]) -> None:
        self.run_func = run_func
        super().__init__(self._name, self._name)

    def add_options(self) -> None:
        index_opts = cmdoptions.make_option_group(cmdoptions.index_group, self.parser)
        self.parser.insert_option_group(0, index_opts)

    def run(self, options: Values, args: List[str]) -> int:
        return self.run_func()


@pytest.fixture
def enable_pip_version_check(monkeypatch: pytest.MonkeyPatch) -> None:
    # The test suite disables the pip self version check by default.
    monkeypatch.setenv("PIP_DISABLE_PIP_VERSION_CHECK", "no")
    # Keep Command.main() from attaching logging handlers to pytest's
    # (short-lived) capture streams, which would leak into later tests.
    monkeypatch.setattr(
        "pip._internal.cli.base_command.setup_logging",
        lambda **kwargs: logging.INFO,
    )


class TestCommand:
    def call_main(self, capsys: pytest.CaptureFixture[str], args: List[str]) -> str:
        """
        Call command.main(), and return the command's stderr.
        """

        def raise_broken_stdout() -> NoReturn:
            raise BrokenStdoutLoggingError()

        cmd = FakeCommand(run_func=raise_broken_stdout)
        status = cmd.main(args)
        assert status == 1
        stderr = capsys.readouterr().err

        return stderr

    def test_raise_broken_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        """
        Test raising BrokenStdoutLoggingError.
        """
        stderr = self.call_main(capsys, [])

        assert stderr.rstrip() == "ERROR: Pipe to stdout was broken"

    def test_raise_broken_stdout__debug_logging(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """
        Test raising BrokenStdoutLoggingError with debug logging enabled.
        """
        stderr = self.call_main(capsys, ["-vv"])

        assert "ERROR: Pipe to stdout was broken" in stderr
        assert "Traceback (most recent call last):" in stderr


@patch("pip._internal.cli.req_command.Command.pip_version_check")
def test_pip_version_check_called(mock_version_check: Mock) -> None:
    """
    Check that ``Command.pip_version_check()`` wraps the command body.
    """
    cmd = FakeCommand()
    cmd.main([])
    mock_version_check.assert_called_once()


@patch("pip._internal.cli.req_command.pip_self_version_check_emit")
@patch("pip._internal.cli.req_command.pip_self_version_check_fetch")
def test_pip_version_check_fetches_before_command_body(
    mock_fetch: Mock, mock_emit: Mock, enable_pip_version_check: None
) -> None:
    """
    The remote pip version is fetched before the command body runs, and the
    upgrade prompt is only emitted after it.
    """
    events: List[str] = []

    def fetch(session: Any, options: Values) -> str:
        events.append("fetch")
        return "2.0"

    def emit(remote_version_str: Optional[str]) -> None:
        events.append("emit")

    mock_fetch.side_effect = fetch
    mock_emit.side_effect = emit

    def run_body() -> int:
        events.append("body")
        return SUCCESS

    cmd = FakeIndexGroupCommand(run_func=run_body)
    assert cmd.main(["--no-cache-dir"]) == SUCCESS

    assert events == ["fetch", "body", "emit"]
    mock_fetch.assert_called_once()
    mock_emit.assert_called_once_with("2.0")


@patch("pip._internal.cli.req_command.pip_self_version_check_emit")
@patch("pip._internal.cli.req_command.pip_self_version_check_fetch")
def test_pip_version_check_fetch_error_does_not_stop_command(
    mock_fetch: Mock,
    mock_emit: Mock,
    enable_pip_version_check: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    An error while fetching the remote pip version is reported as a warning,
    the command body still runs and nothing is emitted afterwards.
    """
    mock_fetch.side_effect = RuntimeError("boom")
    run_body = Mock(return_value=SUCCESS)

    cmd = FakeIndexGroupCommand(run_func=run_body)
    with caplog.at_level(logging.WARNING):
        assert cmd.main(["--no-cache-dir"]) == SUCCESS

    run_body.assert_called_once()
    mock_emit.assert_not_called()
    assert "There was an error checking the latest version of pip." in caplog.messages


@patch("pip._internal.cli.req_command.pip_self_version_check_emit")
@patch("pip._internal.cli.req_command.pip_self_version_check_fetch")
def test_pip_version_check_disabled(
    mock_fetch: Mock, mock_emit: Mock, enable_pip_version_check: None
) -> None:
    """
    Neither phase of the check runs when it is disabled.
    """
    run_body = Mock(return_value=SUCCESS)

    cmd = FakeIndexGroupCommand(run_func=run_body)
    assert cmd.main(["--no-cache-dir", "--disable-pip-version-check"]) == SUCCESS

    run_body.assert_called_once()
    mock_fetch.assert_not_called()
    mock_emit.assert_not_called()


def test_pip_version_check_does_not_run_code_replaced_by_command(
    enable_pip_version_check: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The pip self version check must not run pip code after the command body:
    the command (e.g. ``pip install``) may have overwritten pip's own modules,
    which would then be executed by pip itself (CVE-2026-6357).
    """
    events: List[str] = []

    def get_remote_pip_version(session: Any, options: Values) -> str:
        events.append("fetch")
        return "1.0"

    def overwritten_get_remote_pip_version(*args: Any, **kwargs: Any) -> str:
        events.append("overwritten code executed")
        return "1.0"

    def run_body() -> int:
        events.append("body")
        # Simulate the command replacing the pip code the version check uses.
        monkeypatch.setattr(
            self_outdated_check,
            "_get_current_remote_pip_version",
            overwritten_get_remote_pip_version,
        )
        return SUCCESS

    installed_pip = Mock(version=Version("1.0"), installer="pip")
    monkeypatch.setattr(
        self_outdated_check,
        "get_default_environment",
        lambda: Mock(get_distribution=Mock(return_value=installed_pip)),
    )
    monkeypatch.setattr(
        self_outdated_check, "_get_current_remote_pip_version", get_remote_pip_version
    )

    cmd = FakeIndexGroupCommand(run_func=run_body)
    assert cmd.main(["--no-cache-dir"]) == SUCCESS

    # The remote version is fetched before the command body runs, and nothing
    # replaced by the command body is executed afterwards.
    assert events == ["fetch", "body"]


def test_log_command_success(fixed_time: None, tmpdir: Path) -> None:
    """Test the --log option logs when command succeeds."""
    cmd = FakeCommand()
    log_path = os.path.join(tmpdir, "log")
    cmd.main(["fake", "--log", log_path])
    with open(log_path) as f:
        assert f.read().rstrip() == "2019-01-17T06:00:37,040 fake"


def test_log_command_error(fixed_time: None, tmpdir: Path) -> None:
    """Test the --log option logs when command fails."""
    cmd = FakeCommand(error=True)
    log_path = os.path.join(tmpdir, "log")
    cmd.main(["fake", "--log", log_path])
    with open(log_path) as f:
        assert f.read().startswith("2019-01-17T06:00:37,040 fake")


def test_log_file_command_error(fixed_time: None, tmpdir: Path) -> None:
    """Test the --log-file option logs (when there's an error)."""
    cmd = FakeCommand(error=True)
    log_file_path = os.path.join(tmpdir, "log_file")
    cmd.main(["fake", "--log-file", log_file_path])
    with open(log_file_path) as f:
        assert f.read().startswith("2019-01-17T06:00:37,040 fake")


def test_log_unicode_messages(fixed_time: None, tmpdir: Path) -> None:
    """Tests that logging bytestrings and unicode objects
    don't break logging.
    """
    cmd = FakeCommandWithUnicode()
    log_path = os.path.join(tmpdir, "log")
    cmd.main(["fake_unicode", "--log", log_path])


@pytest.mark.no_auto_tempdir_manager
def test_base_command_provides_tempdir_helpers() -> None:
    assert temp_dir._tempdir_manager is None
    assert temp_dir._tempdir_registry is None

    def assert_helpers_set(options: Values, args: List[str]) -> int:
        assert temp_dir._tempdir_manager is not None
        assert temp_dir._tempdir_registry is not None
        return SUCCESS

    c = Command("fake", "fake")
    # https://github.com/python/mypy/issues/2427
    c.run = Mock(side_effect=assert_helpers_set)  # type: ignore[method-assign]
    assert c.main(["fake"]) == SUCCESS
    c.run.assert_called_once()


not_deleted = "not_deleted"


@pytest.mark.parametrize("kind,exists", [(not_deleted, True), ("deleted", False)])
@pytest.mark.no_auto_tempdir_manager
def test_base_command_global_tempdir_cleanup(kind: str, exists: bool) -> None:
    assert temp_dir._tempdir_manager is None
    assert temp_dir._tempdir_registry is None

    class Holder:
        value: str

    def create_temp_dirs(options: Values, args: List[str]) -> int:
        assert c.tempdir_registry is not None
        c.tempdir_registry.set_delete(not_deleted, False)
        Holder.value = TempDirectory(kind=kind, globally_managed=True).path
        return SUCCESS

    c = Command("fake", "fake")
    # https://github.com/python/mypy/issues/2427
    c.run = Mock(side_effect=create_temp_dirs)  # type: ignore[method-assign]
    assert c.main(["fake"]) == SUCCESS
    c.run.assert_called_once()
    assert os.path.exists(Holder.value) == exists


@pytest.mark.parametrize("kind,exists", [(not_deleted, True), ("deleted", False)])
@pytest.mark.no_auto_tempdir_manager
def test_base_command_local_tempdir_cleanup(kind: str, exists: bool) -> None:
    assert temp_dir._tempdir_manager is None
    assert temp_dir._tempdir_registry is None

    def create_temp_dirs(options: Values, args: List[str]) -> int:
        assert c.tempdir_registry is not None
        c.tempdir_registry.set_delete(not_deleted, False)

        with TempDirectory(kind=kind) as d:
            path = d.path
            assert os.path.exists(path)
        assert os.path.exists(path) == exists
        return SUCCESS

    c = Command("fake", "fake")
    # https://github.com/python/mypy/issues/2427
    c.run = Mock(side_effect=create_temp_dirs)  # type: ignore[method-assign]
    assert c.main(["fake"]) == SUCCESS
    c.run.assert_called_once()
