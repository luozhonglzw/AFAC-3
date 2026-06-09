"""
自动化实验 Agent 框架
核心组件: 配置空间、策略模块、预算管理、停止决策
"""

import os
import json
import time
import random
import numpy as np
from itertools import product
from trajectory_logger import TrajectoryLogger
from experiment_runner import run_cls_experiment, run_rec_experiment


# ============================================================
# 配置空间定义
# ============================================================

CLS_CONFIG_SPACE = {
    "model_type": ["GCN", "GAT", "MLP"],
    "hidden_dim": [128, 256, 512],
    "num_layers": [2, 3, 4],
    "dropout": [0.3, 0.5, 0.7],
    "lr": [0.005, 0.01, 0.02],
    "weight_decay": [1e-4, 5e-4, 1e-3],
    "epochs": [200, 300],
    "patience": [50],
    "symmetrize": [True, False],
    "norm_mode": ["symmetric", "row"],
    "feature_norm": [True, False],
    "use_class_weights": [True, False],
    "num_heads": [4, 8],  # 仅 GAT
    "grad_clip": [5.0],
    "val_ratio": [0.2],
    "seed": [42],
}

REC_CONFIG_SPACE = {
    "embed_dim": [64, 128, 256],
    "num_heads": [2, 4],
    "num_layers": [1, 2, 3],
    "dropout": [0.1, 0.2, 0.3],
    "lr": [5e-4, 1e-3, 2e-3],
    "weight_decay": [1e-5, 1e-4],
    "max_len": [50, 100, 200],
    "batch_size": [128, 256],
    "neg_samples": [50, 100, 200],
    "epochs": [100],
    "patience": [15],
    "seed": [42],
}


# ============================================================
# 策略模块
# ============================================================

class PolicyModule:
    """基于历史实验的策略决策"""

    def __init__(self, config_space, strategy="random_to_greedy"):
        self.config_space = config_space
        self.strategy = strategy
        self.history = []  # [(config, metric)]
        self.phase = "exploration"  # exploration -> exploitation
        self.best_config = None
        self.best_metric = 0

    def suggest_config(self, phase_hint=None):
        """根据策略建议下一组配置"""
        if self.strategy == "random_to_greedy":
            return self._random_to_greedy(phase_hint)
        elif self.strategy == "bayesian":
            return self._bayesian_suggest()
        else:
            return self._random_suggest()

    def update(self, config, metric):
        """更新历史记录"""
        self.history.append((config, metric))
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_config = config.copy()

    def analyze_history(self):
        """从历史实验中归纳规律"""
        if len(self.history) < 3:
            return {"insufficient_data": True}

        metrics = [m for _, m in self.history]
        configs = [c for c, _ in self.history]

        analysis = {
            "total_experiments": len(self.history),
            "best_metric": self.best_metric,
            "metric_mean": np.mean(metrics),
            "metric_std": np.std(metrics),
            "trend": "improving" if metrics[-1] > np.mean(metrics[:len(metrics)//2]) else "stagnating",
        }

        # 分析各超参数的影响
        param_impact = {}
        for key in configs[0]:
            if key in ("seed", "patience", "epochs", "grad_clip", "val_ratio"):
                continue
            values = [c.get(key) for c in configs]
            unique_vals = set(values)
            if len(unique_vals) > 1:
                val_metrics = {}
                for v in unique_vals:
                    idxs = [i for i, c in enumerate(configs) if c.get(key) == v]
                    if idxs:
                        val_metrics[str(v)] = np.mean([metrics[i] for i in idxs])
                if val_metrics:
                    best_val = max(val_metrics, key=val_metrics.get)
                    param_impact[key] = {"best_value": best_val, "all_values": val_metrics}

        analysis["param_impact"] = param_impact

        # 生成建议
        suggestions = []
        if analysis["trend"] == "stagnating":
            suggestions.append("指标停滞，建议扩大搜索范围或切换策略")
        if self.best_config:
            for key, impact in param_impact.items():
                if str(self.best_config.get(key)) != impact["best_value"]:
                    suggestions.append(f"参数 {key} 当前值 {self.best_config.get(key)} 非最优，建议尝试 {impact['best_value']}")

        analysis["suggestions"] = suggestions
        return analysis

    def _random_suggest(self):
        config = {}
        for key, values in self.config_space.items():
            config[key] = random.choice(values)
        return config

    def _random_to_greedy(self, phase_hint=None):
        phase = phase_hint or self.phase

        if phase == "exploration" or len(self.history) < 5:
            return self._random_suggest()

        # 贪心阶段: 基于最优配置微调
        if self.best_config is None:
            return self._random_suggest()

        config = self.best_config.copy()
        # 随机扰动 1-2 个参数
        keys_to_perturb = random.sample(
            [k for k in config if k not in ("seed", "patience", "epochs", "grad_clip", "val_ratio")],
            min(2, len([k for k in config if k not in ("seed", "patience", "epochs", "grad_clip", "val_ratio")]))
        )
        for key in keys_to_perturb:
            if key in self.config_space:
                config[key] = random.choice(self.config_space[key])
        return config

    def _bayesian_suggest(self):
        """简化版贝叶斯: 基于历史选择最优区域"""
        if len(self.history) < 3:
            return self._random_suggest()

        # 取 top-3 实验的配置，取众数
        sorted_hist = sorted(self.history, key=lambda x: -x[1])[:3]
        config = {}
        for key in self.config_space:
            if key in ("seed", "patience", "epochs", "grad_clip", "val_ratio"):
                config[key] = self.config_space[key][0]
                continue
            values = [c.get(key) for c, _ in sorted_hist if key in c]
            if values:
                # 对于数值型，取均值后找最近的
                if isinstance(values[0], (int, float)):
                    mean_val = np.mean(values)
                    config[key] = min(self.config_space[key], key=lambda x: abs(x - mean_val))
                else:
                    # 取众数
                    config[key] = max(set(values), key=values.count)
            else:
                config[key] = random.choice(self.config_space[key])
        return config


# ============================================================
# 预算管理器
# ============================================================

class BudgetManager:
    def __init__(self, max_experiments=30, max_time_hours=2.0, max_gpu_hours=4.0):
        self.max_experiments = max_experiments
        self.max_time_seconds = max_time_hours * 3600
        self.max_gpu_seconds = max_gpu_hours * 3600
        self.start_time = time.time()
        self.experiments_done = 0
        self.total_gpu_time = 0

    def can_continue(self):
        elapsed = time.time() - self.start_time
        if self.experiments_done >= self.max_experiments:
            return False, "达到最大实验次数"
        if elapsed >= self.max_time_seconds:
            return False, "达到最大运行时间"
        return True, "OK"

    def record_experiment(self, duration):
        self.experiments_done += 1
        self.total_gpu_time += duration

    def get_status(self):
        elapsed = time.time() - self.start_time
        return {
            "experiments_done": self.experiments_done,
            "max_experiments": self.max_experiments,
            "elapsed_hours": round(elapsed / 3600, 2),
            "max_hours": round(self.max_time_seconds / 3600, 2),
            "budget_remaining": round(1 - self.experiments_done / self.max_experiments, 2),
        }


# ============================================================
# 停止决策器
# ============================================================

class StopDecider:
    def __init__(self, patience=5, min_improvement=0.005):
        self.patience = patience
        self.min_improvement = min_improvement
        self.no_improve_count = 0
        self.best_metric = 0

    def should_stop(self, current_metric):
        if current_metric > self.best_metric + self.min_improvement:
            self.best_metric = current_metric
            self.no_improve_count = 0
            return False
        self.no_improve_count += 1
        return self.no_improve_count >= self.patience

    def get_reason(self):
        if self.no_improve_count >= self.patience:
            return f"连续 {self.patience} 轮实验无显著提升 (>{self.min_improvement})"
        return None


# ============================================================
# Agent 主框架
# ============================================================

class ExperimentAgent:
    def __init__(self, task_type, data_root, output_dir, budget_config=None):
        """
        task_type: "cls" 或 "rec"
        data_root: 数据根目录
        output_dir: 输出目录 (trajectory 文件)
        """
        self.task_type = task_type
        self.data_root = data_root
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 配置空间
        config_space = CLS_CONFIG_SPACE if task_type == "cls" else REC_CONFIG_SPACE
        self.config_space = config_space

        # 组件
        self.policy = PolicyModule(config_space, strategy="random_to_greedy")
        budget = budget_config or {}
        self.budget = BudgetManager(
            max_experiments=budget.get("max_experiments", 30),
            max_time_hours=budget.get("max_time_hours", 2.0),
        )
        self.stop_decider = StopDecider(
            patience=budget.get("stop_patience", 5),
            min_improvement=budget.get("min_improvement", 0.005),
        )
        task_name = "B1" if task_type == "cls" else "B2"
        self.logger = TrajectoryLogger(output_dir, task_name)

        # 状态
        self.best_result = None
        self.all_results = []

    def run(self):
        """运行完整的自动实验流程"""
        print(f"\n{'='*60}")
        print(f"  自动化实验 Agent - 任务 {'分类' if self.task_type == 'cls' else '推荐'}")
        print(f"{'='*60}")
        print(f"  预算: 最多 {self.budget.max_experiments} 轮实验")
        print(f"  策略: {self.policy.strategy}")
        print(f"  设备: {self.data_root}")
        print()

        round_id = 0
        while True:
            # 检查预算
            can_run, reason = self.budget.can_continue()
            if not can_run:
                print(f"\n[Agent] 停止: {reason}")
                break

            # 检查停止条件
            if self.stop_decider.should_stop(
                self.best_result["metrics"]["primary_metric"] if self.best_result else 0
            ):
                print(f"\n[Agent] 停止: {self.stop_decider.get_reason()}")
                break

            round_id += 1
            print(f"\n--- 实验轮次 {round_id} ---")

            # 策略建议配置
            # 根据历史调整策略阶段
            if len(self.policy.history) >= 8:
                self.policy.phase = "exploitation"
            config = self.policy.suggest_config()

            # 过滤无效配置
            config = self._filter_config(config)
            print(f"  配置: {self._config_summary(config)}")

            # 执行实验
            try:
                if self.task_type == "cls":
                    result = run_cls_experiment(config, self.data_root)
                else:
                    result = run_rec_experiment(config, self.data_root)
            except Exception as e:
                print(f"  [错误] 实验失败: {e}")
                self.logger.log_experiment(round_id, config, {}, {"error": str(e)}, 0)
                continue

            # 记录结果
            metric = result["metrics"]["primary_metric"]
            self.policy.update(config, metric)
            self.budget.record_experiment(result["duration"])
            self.all_results.append(result)

            # 更新最优
            is_best = False
            if self.best_result is None or metric > self.best_result["metrics"]["primary_metric"]:
                self.best_result = result
                is_best = True

            # 日志
            self.logger.log_experiment(
                round_id, config, result["metrics"], result["feedback"], result["duration"]
            )
            self.logger.log_decision(
                round_id,
                "config_selected",
                f"策略: {self.policy.strategy}, 阶段: {self.policy.phase}",
                {"is_best": is_best}
            )

            print(f"  指标: {metric:.4f} {'[最优]' if is_best else ''}")
            print(f"  耗时: {result['duration']:.1f}s")
            print(f"  反馈: {result['feedback']}")

        # 输出最终结果
        self._finalize()
        return self.best_result

    def _filter_config(self, config):
        """过滤不兼容的配置组合"""
        if config.get("model_type") != "GAT":
            config.pop("num_heads", None)
        return config

    def _config_summary(self, config):
        """简要显示配置"""
        keys = ["model_type", "hidden_dim", "num_layers", "dropout", "lr", "embed_dim"]
        return {k: config[k] for k in keys if k in config}

    def _finalize(self):
        """输出最终结果和分析"""
        print(f"\n{'='*60}")
        print(f"  实验结束 - 最终结果")
        print(f"{'='*60}")

        if self.best_result:
            print(f"  最优指标: {self.best_result['metrics']['primary_metric']:.4f}")
            print(f"  最优配置: {self.best_result['config']}")

        # 分析
        analysis = self.policy.analyze_history()
        print(f"\n  实验分析:")
        print(f"    总实验数: {analysis.get('total_experiments', 0)}")
        print(f"    指标均值: {analysis.get('metric_mean', 0):.4f}")
        print(f"    指标标准差: {analysis.get('metric_std', 0):.4f}")
        print(f"    趋势: {analysis.get('trend', 'N/A')}")

        if analysis.get("suggestions"):
            print(f"\n  策略建议:")
            for s in analysis["suggestions"]:
                print(f"    - {s}")

        # 预算状态
        budget_status = self.budget.get_status()
        print(f"\n  预算使用: {budget_status['experiments_done']}/{budget_status['max_experiments']} 轮")

        # 保存摘要
        summary = self.logger.get_summary()
        summary["analysis"] = analysis
        summary["budget_status"] = budget_status
        summary_path = os.path.join(self.output_dir, f"summary_{'B1' if self.task_type == 'cls' else 'B2'}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n  摘要已保存至 {summary_path}")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="自动化实验 Agent")
    parser.add_argument("--task", type=str, choices=["cls", "rec", "both"], default="both")
    parser.add_argument("--data_root", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--max_experiments", type=int, default=20)
    parser.add_argument("--max_time_hours", type=float, default=2.0)
    args = parser.parse_args()

    budget_config = {
        "max_experiments": args.max_experiments,
        "max_time_hours": args.max_time_hours,
        "stop_patience": 5,
        "min_improvement": 0.005,
    }

    if args.task in ("cls", "both"):
        agent = ExperimentAgent("cls", args.data_root, args.output_dir, budget_config)
        agent.run()

    if args.task in ("rec", "both"):
        agent = ExperimentAgent("rec", args.data_root, args.output_dir, budget_config)
        agent.run()
