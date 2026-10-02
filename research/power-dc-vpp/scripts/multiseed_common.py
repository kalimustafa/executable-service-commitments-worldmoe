"""Shared immutable design and lightweight progress for the approved study."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'artifacts/r_ecmoe_multiseed_20260910'
SEEDS = (20240813, 20240829, 20240917)
W0_BUDGETS = {20240813: 128, 20240829: 96, 20240917: 96}
VARIANTS = {
    'full': '完整模型',
    'no_sparse_coordinator': '无稀疏协调器',
    'no_risk_condition': '无风险条件',
    'no_decision_tail': '无决策尾部',
    'no_route_prior': '无路由先验',
    'pooled_latent_instead_of_expert_states': '池化表示替代专家状态',
    'tail_always_on': '尾部始终开启',
}

def now():
    return datetime.now(timezone.utc).isoformat()

def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return default

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')
    tmp.replace(path)

def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('Cannot write an empty evidence table')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def world_run(seed):
    return f'multiseed-20260910-seed{seed}'

def world_phase(seed, phase):
    return ROOT/'world_moe'/f'run={world_run(seed)}'/f'phase={phase}'

def r_run(seed, variant):
    return f'multiseed-20260910-seed{seed}-{variant}'

def r_root(seed, variant):
    return ROOT/'r'/r_run(seed, variant)

def config_path(seed, phase, variant=None):
    suffix = phase if variant is None else f'r_{variant}'
    return REPO/'configs'/f'multiseed_20260910_{seed}_{suffix}.yaml'

def jobs():
    result = []
    for seed in SEEDS:
        for phase in ('w0','w1'):
            result.append({'id':f'{seed}/{phase}', 'seed':seed, 'phase':phase,
                           'variant':None, 'run_root':str(world_phase(seed,phase)),
                           'config':str(config_path(seed,phase)), 'status':'pending'})
        for variant in VARIANTS:
            result.append({'id':f'{seed}/{variant}', 'seed':seed, 'phase':'r',
                           'variant':variant, 'run_root':str(r_root(seed,variant)),
                           'config':str(config_path(seed,'r',variant)), 'status':'pending'})
    return result

def render_plan(state):
    completed = sum(j['status']=='completed' for j in state['jobs'])
    lines = ['# 三种子独立消融执行计划', '',
             f'状态：**{state["status"]}**；训练完成 **{completed}/27**；更新时间：{state["updated_at"]}。', '',
             '种子：20240813、20240829、20240917。完整模型维持Top-2，不安排Top-1/Top-2对照。', '',
             '| 种子 | 阶段/配置 | 状态 | 轮数 | 更新数 | 已用时间(秒) |',
             '|---|---|---|---:|---:|---:|']
    for j in state['jobs']:
        progress = j.get('progress', {}) or {}
        label = VARIANTS.get(j.get('variant'), str(j['phase']).upper())
        epoch = progress.get('epoch',progress.get('short_epoch','—'))
        update = progress.get('global_update',progress.get('updates',progress.get('update','—')))
        lines.append(f'| {j["seed"]} | {label} | {j["status"]} | {epoch} | {update} | {j.get("elapsed_seconds",0):.0f} |')
    phases = [('implementation_verified','训练调度与消融分支检查'),
              ('models_frozen','冻结21个R及对应W1'), ('evaluation_completed','完成2025：21组×362天×6窗口'),
              ('report_completed','完成报告、逐种子结果、跨种子统计与效率分析')]
    lines += ['','## 后续完成项','']
    for key, label in phases:
        lines.append(f'- [{"x" if state.get(key) else " "}] {label}')
    lines += ['', '## 固定执行规则', '',
              '- W0：种子20240813已完成128短轮；20240829与20240917均为96短轮（12遍/732更新）。首次val32，此后每16。',
              '- W1：前47轮不val，第48轮首次，此后每4轮；建立最优点后连续2次未刷新最优即停，无独立起算轮数或最小改善百分比，上限80轮。',
              '- R：前77轮不val，第78轮首次，此后每13轮；建立最优点后连续2次未刷新最优即停，无独立起算轮数或最小改善百分比，上限208轮。',
              '- 训练2022–2023，选模2024；全部冻结后测试2025。',
              '- 主指标同时覆盖平均成本、CVaR95、SLA及物理约束。',
              '- 按用户修订采用validation_score_only：按正式验证评分选最优，科学阈值只作诊断，不阻止阶段衔接。',
              '- 首次正式val前不自动保存模型；此后仅刷新最优评分时写一次，取消定时和未改善节点保存。手动暂停/异常仍保存续训点。',
              '- 阶段完成后只保留最优模型，兼容文件名可指向同一权重；已执行的历史验证和保存按实测日志记录。',
              '- 数据、模型、配置来源写入对应manifest；状态来自实际进程与输出。']
    if state.get('error'):
        lines += ['', '## 当前阻塞', '', str(state['error'])]
    path = ROOT/'PLAN.md'; path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix('.md.tmp'); tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8'); tmp.replace(path)

def publish(state):
    state['updated_at']=now()
    write_json(ROOT/'study_status.json',state)
    render_plan(state)
