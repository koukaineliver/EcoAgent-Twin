from importlib.util import find_spec
from threading import Event, Thread, get_ident
from time import perf_counter, sleep
import unittest

from ecoagent_twin.m5_deadline import DeadlineModel, WorkerBusy, WorkerUnavailable


class _Backend:
    def __init__(self):
        self.last_scores = {}
        self.calls = []

    def __call__(self, view, *, timeout_s):
        self.calls.append((view, timeout_s, get_ident()))
        self.last_scores = {view: len(self.calls)}
        return {"candidate_id": view}


class DeadlineTest(unittest.TestCase):
    def test_loads_and_primes_once_and_isolates_accepted_scores(self):
        backend = _Backend()
        factory_threads = []

        def factory():
            factory_threads.append(get_ident())
            return backend

        with DeadlineModel(factory) as model:
            model.warmup("prime", timeout_s=2)
            self.assertEqual(model.last_status, "ready")
            self.assertEqual(model.last_scores, {})
            self.assertEqual(model("first", timeout_s=1), {"candidate_id": "first"})
            self.assertEqual(model.last_status, "completed")
            self.assertEqual(model.accepted_scores({"status": "selected"}), {"first": 2})
            scores = model.last_scores
            scores.clear()
            self.assertEqual(model.last_scores, {"first": 2})
            self.assertEqual(model.accepted_scores({"status": "timeout", "fallback_used": True}), {})
            self.assertEqual(model("second", timeout_s=1), {"candidate_id": "second"})
            self.assertEqual(model.last_scores, {"second": 3})
        self.assertEqual(len(factory_threads), 1)
        self.assertNotEqual(factory_threads[0], get_ident())
        self.assertTrue(all(thread == factory_threads[0] for _, _, thread in backend.calls))
        self.assertTrue(all(0 < timeout < 1 for _, timeout, _ in backend.calls[1:]))

    def test_late_inference_exposes_no_scores_and_creates_no_backlog(self):
        release = Event()
        started = Event()
        backend = _Backend()

        def blocked(view, *, timeout_s):
            if view == "late":
                started.set()
                release.wait(3)
            proposal = backend(view, timeout_s=timeout_s)
            blocked.last_scores = backend.last_scores
            return proposal

        model = DeadlineModel(lambda: blocked, reserve_s=0.1)
        try:
            model.warmup()
            before = perf_counter()
            with self.assertRaises(TimeoutError):
                model("late", timeout_s=0.25)
            self.assertTrue(started.is_set())
            self.assertLess(perf_counter() - before, 0.65)
            self.assertEqual(model.last_status, "timeout")
            self.assertEqual(model.last_scores, {})
            with self.assertRaises(WorkerBusy):
                model("must-not-queue", timeout_s=1)
            self.assertEqual(model.last_status, "busy")
            self.assertEqual(model.last_scores, {})

            release.set()
            for _ in range(100):
                try:
                    self.assertEqual(model("fresh", timeout_s=1), {"candidate_id": "fresh"})
                    break
                except WorkerBusy:
                    sleep(0.005)
            else:
                self.fail("worker did not become idle after releasing its old inference")
            self.assertEqual([view for view, _, _ in backend.calls], ["late", "fresh"])
            self.assertEqual(model.last_scores, {"fresh": 2})
        finally:
            release.set()
            model.close(wait_s=1)

    def test_busy_rejection_does_not_republish_an_earlier_request(self):
        release = Event()
        entered = Event()
        output = []

        def backend(view, *, timeout_s):
            entered.set()
            release.wait(2)
            backend.last_scores = {"first": 8.0}
            return "first"

        with DeadlineModel(lambda: backend) as model:
            model.warmup()
            caller = Thread(target=lambda: output.append(model("first", timeout_s=1.5)))
            caller.start()
            try:
                self.assertTrue(entered.wait(1))
                with self.assertRaises(WorkerBusy):
                    model("second", timeout_s=1)
                self.assertEqual(model.last_status, "busy")
            finally:
                release.set()
                caller.join(2)
            self.assertEqual(output, ["first"])
            self.assertEqual(model.last_scores, {})
            self.assertEqual(model.last_status, "busy")

    def test_reserve_shutdown_and_failed_initialization(self):
        backend = _Backend()
        model = DeadlineModel(lambda: backend, reserve_s=0.1)
        model.warmup()
        with self.assertRaises(TimeoutError):
            model("forbidden", timeout_s=0.1)
        self.assertEqual(backend.calls, [])
        self.assertTrue(model.close(wait_s=1))
        with self.assertRaises(WorkerUnavailable):
            model("closed", timeout_s=1)
        self.assertEqual(model.last_scores, {})
        self.assertEqual(model.last_status, "unavailable")

        def fail():
            raise ValueError("unloadable snapshot")

        with DeadlineModel(fail) as unavailable:
            with self.assertRaises(WorkerUnavailable):
                unavailable.warmup()
            self.assertEqual(unavailable.last_status, "unavailable")

    @unittest.skipUnless(find_spec("langgraph"), "LangGraph is required for fallback integration")
    def test_langgraph_dispatches_fallback_before_running_backend_completes(self):
        from ecoagent_twin.m3_supervisor import Candidate, _graph, supervise

        release = Event()
        entered = Event()
        _graph()

        def backend(view, *, timeout_s):
            entered.set()
            release.wait(3)
            backend.last_scores = {"continue_mappo": 3.0}
            return "continue_mappo"

        with DeadlineModel(lambda: backend, reserve_s=0.1) as model:
            model.warmup()
            menu = (
                Candidate("continue_mappo", "Continue", 2, 800, 2, 22),
                Candidate("route_repair", "Repair", 3, 850, 1, 22),
            )
            try:
                receipt = supervise({"kind": "outage"}, menu, model=model, budget_s=0.5)
                self.assertTrue(entered.is_set())
                self.assertEqual(receipt["status"], "timeout")
                self.assertEqual(receipt["decision_id"], "route_repair")
                self.assertFalse(receipt["deadline_exceeded"])
                self.assertEqual(model.accepted_scores(receipt), {})
            finally:
                release.set()


if __name__ == "__main__":
    unittest.main()
