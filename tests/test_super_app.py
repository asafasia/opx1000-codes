import unittest
import threading
import os
import subprocess
from unittest.mock import patch

from apps.super_app.desktop import DesktopRuntime, main as desktop_main
from apps.super_app.server import (
    APPS,
    AppManager,
    PROJECT_ROOT,
    read_wiki_document,
    resolve_wiki_path,
    wiki_index,
)


class SuperAppTests(unittest.TestCase):
    def test_hub_contains_all_current_lab_apps(self):
        self.assertEqual(
            [(app.name, app.port) for app in APPS],
            [
                ("Data Review", 8892),
                ("Lab Monitor", 8895),
                ("Fridge Monitor", 8765),
                ("Oscilloscope", 8851),
                ("Profile Studio", 8893),
                ("Parameter Sweep", 8770),
            ],
        )
        self.assertTrue(all(app.server_path.is_file() for app in APPS))
        self.assertEqual(PROJECT_ROOT.name, "opx1000-codes")

    def test_every_surface_uses_one_shared_horizontal_navigation(self):
        pages = {
            "overview": PROJECT_ROOT / "apps" / "super_app" / "static" / "index.html",
            "wiki": PROJECT_ROOT / "apps" / "super_app" / "static" / "wiki.html",
            "data-review": PROJECT_ROOT / "apps" / "visualiser" / "static" / "index.html",
            "lab-monitor": PROJECT_ROOT / "apps" / "job_status" / "static" / "index.html",
            "fridge-monitor": PROJECT_ROOT / "apps" / "super_app" / "static" / "fridge.html",
            "oscilloscope": PROJECT_ROOT / "apps" / "super_app" / "static" / "oscilloscope.html",
            "profile-studio": PROJECT_ROOT / "apps" / "profile_studio" / "static" / "index.html",
            "parameter-sweep": PROJECT_ROOT / "apps" / "parameter_scan" / "static" / "index.html",
        }

        for tab_id, page_path in pages.items():
            page = page_path.read_text(encoding="utf-8")
            self.assertIn(f'data-lab-tab="{tab_id}"', page)
            self.assertIn("lab-tabs.js?v=4", page)
            self.assertIn("lab-tabs.css?v=5", page)

        navigation = (
            PROJECT_ROOT / "apps" / "super_app" / "static" / "lab-tabs.js"
        ).read_text(encoding="utf-8")
        self.assertIn("Quantum coherence lab", navigation)
        for label in (
            "Overview",
            "Data Review",
            "Lab Monitor",
            "Fridge Monitor",
            "Oscilloscope",
            "Profile Studio",
            "Parameter Sweep",
            "Wiki",
        ):
            self.assertIn(f'label: "{label}"', navigation)

    @patch("apps.super_app.server.probe_app", return_value=(True, None))
    def test_status_reports_existing_apps_without_claiming_management(self, _probe):
        manager = AppManager("127.0.0.1")

        status = manager.status(APPS[0])

        self.assertTrue(status["running"])
        self.assertFalse(status["managed"])

    @patch("apps.super_app.server.subprocess.Popen")
    @patch("apps.super_app.server.probe_app", return_value=(True, None))
    def test_start_all_leaves_existing_servers_alone(self, _probe, popen):
        manager = AppManager("127.0.0.1")

        manager.start_all()
        manager.stop_all()

        popen.assert_not_called()

    @patch("apps.super_app.server.port_is_open", return_value=True)
    @patch(
        "apps.super_app.server.probe_app",
        return_value=(False, "Port 8892 is serving a different application."),
    )
    def test_start_does_not_overwrite_a_foreign_listener(self, _probe, _port):
        manager = AppManager("127.0.0.1")

        manager._start(APPS[0])

        self.assertNotIn(APPS[0].id, manager.processes)
        self.assertIn("different application", manager.last_errors[APPS[0].id])

    def test_main_binds_hub_before_starting_children(self):
        calls = []

        class FakeServer:
            def __init__(self, _address, _manager):
                calls.append("bind")

            def serve_forever(self):
                calls.append("serve")

            def server_close(self):
                calls.append("close")

        with patch("apps.super_app.server.SuperAppServer", FakeServer), patch.object(
            AppManager, "start_all", side_effect=lambda: calls.append("start")
        ), patch.object(AppManager, "stop_all", side_effect=lambda: calls.append("stop")), patch(
            "sys.argv", ["server.py", "--no-launch"]
        ):
            from apps.super_app.server import main

            main()

        self.assertLess(calls.index("bind"), calls.index("start"))

    def test_wiki_discovers_and_reads_repository_markdown(self):
        index = wiki_index()

        self.assertGreaterEqual(index["count"], 20)
        self.assertIn("README.md", [item["path"] for item in index["documents"]])
        document = read_wiki_document("README.md")
        self.assertEqual(document["path"], "README.md")
        self.assertIn("OPX1000", document["content"])

    def test_wiki_searches_document_contents(self):
        results = wiki_index("hardware")

        self.assertGreater(results["count"], 0)
        self.assertTrue(any("hardware" in item["path"].lower() for item in results["documents"]))

    def test_wiki_rejects_paths_outside_repository(self):
        with self.assertRaises(PermissionError):
            resolve_wiki_path("../secret.md")

    def test_desktop_runtime_owns_hub_lifecycle(self):
        served = threading.Event()

        class FakeServer:
            def __init__(self, address, manager):
                self.address = address
                self.manager = manager
                self.shutdown_called = False
                self.closed = False

            def serve_forever(self):
                served.set()

            def shutdown(self):
                self.shutdown_called = True

            def server_close(self):
                self.closed = True

        runtime = DesktopRuntime("127.0.0.1", 8890)
        with patch("apps.super_app.desktop.probe_existing_hub", return_value=(False, "offline")), patch(
            "apps.super_app.desktop.port_is_open", return_value=False
        ), patch("apps.super_app.desktop.SuperAppServer", FakeServer), patch.object(
            runtime.manager, "start_all"
        ) as start_all, patch.object(runtime.manager, "stop_all") as stop_all:
            runtime.start()
            self.assertTrue(served.wait(timeout=1))
            if runtime.services_thread is not None:
                runtime.services_thread.join(timeout=1)
            runtime.stop()
            runtime.stop()

        start_all.assert_called_once_with()
        stop_all.assert_called_once_with()
        self.assertEqual(runtime.url, "http://127.0.0.1:8890/?desktop=1")
        self.assertTrue(runtime.server.closed)

    @unittest.skipUnless(os.name == "nt", "Windows-only process visibility behavior")
    @patch("apps.super_app.server.subprocess.Popen")
    @patch("apps.super_app.server.port_is_open", return_value=False)
    @patch("apps.super_app.server.probe_app", return_value=(False, "offline"))
    def test_linked_services_start_without_console_windows(self, _probe, _port, popen):
        process = popen.return_value
        process.poll.return_value = None
        manager = AppManager("127.0.0.1")

        manager._start(APPS[0])
        options = popen.call_args.kwargs

        self.assertTrue(options["creationflags"] & subprocess.CREATE_NO_WINDOW)
        self.assertTrue(options["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP)
        self.assertTrue(options["startupinfo"].dwFlags & subprocess.STARTF_USESHOWWINDOW)
        manager.stop_all()

    def test_desktop_runtime_reuses_verified_existing_hub(self):
        runtime = DesktopRuntime("127.0.0.1", 8890)
        with patch("apps.super_app.desktop.probe_existing_hub", return_value=(True, None)), patch.object(
            runtime.manager, "start_all"
        ) as start_all, patch.object(runtime.manager, "stop_all") as stop_all:
            runtime.start()
            runtime.stop()

        self.assertTrue(runtime.using_existing_hub)
        self.assertIsNone(runtime.server)
        start_all.assert_not_called()
        stop_all.assert_called_once_with()

    def test_desktop_main_opens_native_edge_window(self):
        calls = []
        callbacks = []

        class FakeEvents:
            def __init__(self):
                self.closed = self

            def __iadd__(self, callback):
                calls.append(("closed", callback))
                callbacks.append(callback)
                return self

        class FakeWebview:
            @staticmethod
            def create_window(title, url, **options):
                calls.append(("window", title, url, options))
                return type("FakeWindow", (), {"events": FakeEvents()})()

            @staticmethod
            def start(**options):
                calls.append(("start", options))
                callbacks[0]()

        with patch("apps.super_app.desktop.load_webview", return_value=FakeWebview), patch.object(
            DesktopRuntime, "start"
        ), patch.object(DesktopRuntime, "stop"):
            result = desktop_main(["--no-launch"])

        self.assertEqual(result, 0)
        self.assertEqual(
            calls[0][0:3],
            (
                "window",
                "OPX1000 Quantum Coherence Lab",
                "http://127.0.0.1:8890/?desktop=1",
            ),
        )
        self.assertEqual(calls[-1], ("start", {"gui": "edgechromium", "debug": False}))


if __name__ == "__main__":
    unittest.main()
