import abc
import logging
import os
import time
from concurrent.futures.thread import ThreadPoolExecutor

from loky import get_reusable_executor

from aikit.automl.launcher import AutoMlLauncher
from aikit.automl.persistence.reader import ResultsReader
from aikit.automl.persistence.storage import Storage
from aikit.automl.utils import deactivate_warnings, configure_console_logging
from aikit.datasets import load_dataset, DatasetEnum
from aikit.model_definition import sklearn_model_from_param
from aikit.tools import save_pkl


class AikitEstimator(metaclass=abc.ABCMeta):

    def __init__(self, max_model_count=None,
                 max_runtime_seconds=None,
                 n_jobs=1,
                 path='./aikit_workdir'):
        self.path = path
        # TODO: move in storage layer
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.storage = Storage(self.path)
        self.max_model_count = max_model_count
        self.n_jobs = n_jobs
        self.max_runtime_seconds = max_runtime_seconds
        self.timeout = self.max_runtime_seconds * 1.2

    def _run_automl(self, X, y, ):
        def _start_controller():
            try:
                AutoMlLauncher(self.path).start_controller(
                    max_runtime_seconds=self.max_runtime_seconds,
                    max_model_count=self.max_model_count)
            except Exception as e:
                print(e)

        def _start_worker():
            logging.getLogger('aikit').info(
                f"Worker started in process {os.getpid()}.")
            AutoMlLauncher(self.path).start_worker()

        save_pkl((X, y), os.path.join(self.path, 'data.pkl'))

        # Spawn controller
        controller = ThreadPoolExecutor(max_workers=1)
        future1 = controller.submit(_start_controller)
        time.sleep(1)

        # Spawn worker processes
        if self.n_jobs > 1:
            worker = get_reusable_executor(max_workers=self.n_jobs, timeout=2)
            futures = [worker.submit(_start_worker) for _ in range(self.n_jobs)]
        else:
            worker = ThreadPoolExecutor(max_workers=1)
            futures = [worker.submit(_start_worker)]

        # Wait all controller/workers to finish
        time.sleep(5)
        future1.result()
        logging.getLogger("aikit").info("Stop all workers...")
        AutoMlLauncher(self.path).stop_workers()
        for f in futures:
            f.result()


class AikitClassifier(AikitEstimator):

    def __init__(self, main_scorer='accuracy', n_jobs=1, scorers=None, max_model_count=None,
                 max_runtime_seconds=600, seed=123, path='.'):
        super().__init__(max_model_count, max_runtime_seconds, n_jobs, path)
        self.seed = seed
        self.main_scorer = main_scorer
        self.scorers = scorers
        if self.scorers is None:
            self.scorers = ['accuracy', 'log_loss_patched', 'avg_roc_auc', 'f1_macro']
        self.best_model_ = None
        self.models_ = []

    def _load_model(self, job_id):
        job = self.storage.load_special_json(job_id, 'jobs')
        return sklearn_model_from_param(job['model_code'])

    def fit(self, X, y, **kwargs):
        self._run_automl(X, y)

        df = ResultsReader(self.storage).get_results(agregate=True)
        best_result = df.sort_values(['test_'+self.main_scorer])[::-1].iloc[0]
        best_job_id = best_result['job_id']
        best_metric = best_result['test_'+self.main_scorer]
        logging.getLogger('aikit').info(f"Best model: {best_job_id}, {self.main_scorer}={best_metric}")

        self.best_model_ = self._load_model(best_job_id)
        self.best_model_.fit(X, y)

        self.models_ = [self._load_model(x) for x in df['job_id'].unique()]

        return self

    def predict(self, X):
        return self.best_model_.predict(X)

    def predict_proba(self, X):
        return self.best_model_.predict_proba(X)


if __name__ == "__main__":
    deactivate_warnings()
    configure_console_logging()
    X, y, *_ = load_dataset(DatasetEnum.titanic)
    estimator = AikitClassifier(max_model_count=1, path='./automl_workdir')
    estimator.fit(X, y)
