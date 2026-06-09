"""
轨迹日志记录器 - 记录每轮实验的完整信息
输出 trajectory_B1.json / trajectory_B2.json
"""

import json
import time
import os
from datetime import datetime


class TrajectoryLogger:
    def __init__(self, output_dir, task_name):
        self.output_dir = output_dir
        self.task_name = task_name
        self.trajectory = []
        self.start_time = time.time()

    def log_experiment(self, exp_id, config, metrics, feedback, duration):
        """记录一轮实验"""
        entry = {
            "experiment_id": exp_id,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(time.time() - self.start_time, 2),
            "config": config,
            "metrics": metrics,
            "feedback": feedback,
            "duration_seconds": round(duration, 2),
        }
        self.trajectory.append(entry)

        # 实时追加写入
        self._save()
        return entry

    def log_decision(self, exp_id, decision_type, reason, details=None):
        """记录决策过程"""
        entry = {
            "experiment_id": exp_id,
            "timestamp": datetime.now().isoformat(),
            "decision_type": decision_type,
            "reason": reason,
            "details": details or {},
        }
        # 附加到最近的实验记录
        if self.trajectory:
            if "decisions" not in self.trajectory[-1]:
                self.trajectory[-1]["decisions"] = []
            self.trajectory[-1]["decisions"].append(entry)
            self._save()

    def get_summary(self):
        """生成轨迹摘要"""
        if not self.trajectory:
            return {"total_experiments": 0}

        valid_exps = [e for e in self.trajectory if "metrics" in e and e["metrics"]]
        if not valid_exps:
            return {"total_experiments": len(self.trajectory)}

        best_exp = max(valid_exps, key=lambda e: e["metrics"].get("primary_metric", 0))
        return {
            "total_experiments": len(self.trajectory),
            "valid_experiments": len(valid_exps),
            "best_experiment_id": best_exp["experiment_id"],
            "best_config": best_exp["config"],
            "best_metrics": best_exp["metrics"],
            "total_duration_seconds": round(time.time() - self.start_time, 2),
        }

    def _save(self):
        path = os.path.join(self.output_dir, f"trajectory_{self.task_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.trajectory, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.trajectory = json.load(f)
