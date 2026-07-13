import unittest
from unittest.mock import patch

from apps.super_app.server import (
    APPS,
    AppManager,
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
                ("Profile Studio", 8893),
                ("Parameter Sweep", 8770),
            ],
        )
        self.assertTrue(all(app.server_path.is_file() for app in APPS))

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


if __name__ == "__main__":
    unittest.main()
