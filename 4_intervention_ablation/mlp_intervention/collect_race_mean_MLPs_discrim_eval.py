#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect race-specific mean MLP activations (last token) for discrim-eval.

注意：根据用户要求，这里 **不单独在 discrim-eval 上求均值**，
而是仅作为一个占位/检查脚本，提示应直接复用在 Resume 样本上收集的均值。

在一键脚本中，推荐做法是：
- 使用 exp15/collect_race_mean_MLPs_resume.py 在 Resume 数据上收集均值；
- 在 discrim-eval 干预脚本中，直接加载该 Resume 均值文件。

因此本脚本仅保留一个明确的报错信息，避免误调用。
"""

import sys


def main() -> None:
    raise RuntimeError(
        "collect_race_mean_MLPs_discrim_eval.py is intentionally a stub. "
        "All MLP mean-ablation interventions should use means collected on the Resume dataset "
        "via collect_race_mean_MLPs_resume.py, as per the experiment design."
    )


if __name__ == "__main__":
    main()
