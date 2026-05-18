import os
import json
import csv as csv_mod
import argparse
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

METHOD_MAP = {
    'context-aware': 'lr',
    'dr_adjusted_mean': 'dr_adjusted_mean',
}


# ============================================================
# 1. 配置加载
# ============================================================
def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_registry = cfg.get('MODEL_REGISTRY', {})
    method_name = cfg.get('method', 'context-aware')
    method = METHOD_MAP.get(method_name, method_name)
    if method not in ('lr', 'dr_adjusted_mean'):
        raise ValueError(f"不支持的方法: {method_name} (→ {method}), 目前仅支持: context-aware, dr_adjusted_mean")
    return model_registry, method


# ============================================================
# 2. 数据加载
# ============================================================
DEFECT_RATES = {
    '__default__': 0.2899116330309636,
    'book_other': 0.3161290322580645,
    'book_paper': 0.35795187990081095,
    'book_plastic_tight_wrap': 0.24135220125786164,
    'cardboard': 0.3663216653099198,
    'other': 0.17475728155339806,
    'paper': 0.4822761194029851,
    'plastic_bubble_wrap': 0.16722595078299776,
    'plastic_hard': 0.1301041301041301,
    'plastic_loose_bag': 0.26766946292546523,
    'plastic_tight_wrap': 0.1958844479620103,
}


def fuse_dr_adjusted_mean(X_raw, mat_arr, dr_map):
    global_dr = dr_map.get('__default__', 0.1)
    dr = np.array([dr_map.get(m, dr_map.get('__default__', 0.1)) for m in mat_arr])
    ratio = dr / global_dr
    X_adj = X_raw * ratio[:, np.newaxis]
    X_adj = np.clip(X_adj, 0, 1)
    return X_adj.mean(axis=1)


def load_preds(filepath):
    with open(filepath) as f:
        data = json.load(f)
    preds = {}
    for item in data:
        cid = item.get('capture_id', '')
        p = item.get('pred', 0)
        try:
            p = float(p)
        except Exception:
            p = 0.0
        preds[cid] = max(0.0, min(1.0, p))
    return preds


def load_preds_with_mat(filepath):
    with open(filepath) as f:
        data = json.load(f)
    preds, mat = {}, {}
    for item in data:
        cid = item.get('capture_id')
        p = item.get('pred')
        try:
            p = float(p)
        except Exception:
            p = 0.0
        preds[cid] = max(0.0, min(1.0, p))
        mat[cid] = item.get('item_material')
    return preds, mat


def load_gt_and_mat(filepath):
    with open(filepath) as f:
        data = json.load(f)
    gt, mat = {}, {}
    for item in data:
        cid = item.get('capture_id', '')
        defect = item.get('defect', item.get('gt_is_defect', None))
        if defect is not None:
            gt[cid] = bool(defect)
        mat[cid] = item.get('item_material', '')
    return gt, mat


# ============================================================
# 3. 特征工程
# ============================================================
def build_features(X_raw, mat_arr, le, dr_map, model_names, use_hierarchical=True):
    n_mod = len(model_names)
    known = set(le.classes_)
    default = len(le.classes_)

    mat_enc = np.array([le.transform([m])[0] if m in known else default
                        for m in mat_arr]).reshape(-1, 1)
    dr = np.array([dr_map.get(m, dr_map.get('__default__', 0))
                   for m in mat_arr]).reshape(-1, 1)

    pred_mean = X_raw.mean(axis=1).reshape(-1, 1)
    pred_std = X_raw.std(axis=1).reshape(-1, 1)
    pred_max = X_raw.max(axis=1).reshape(-1, 1)
    pred_min = X_raw.min(axis=1).reshape(-1, 1)
    pred_range = (pred_max - pred_min)

    ranks = np.argsort(np.argsort(X_raw, axis=1), axis=1)
    rank_std = ranks.std(axis=1).reshape(-1, 1)
    rank_range = (ranks.max(axis=1) - ranks.min(axis=1)).reshape(-1, 1)

    interactions = [(X_raw[:, i] * dr.ravel()).reshape(-1, 1) for i in range(n_mod)]

    feats = [X_raw, mat_enc,
             pred_mean, pred_std, pred_max, pred_min, pred_range,
             rank_std, rank_range]
    feats.extend(interactions)

    if use_hierarchical:
        is_plastic = np.array([1.0 if 'plastic' in m.lower() else 0.0
                               for m in mat_arr]).reshape(-1, 1)
        is_book = np.array([1.0 if 'book' in m.lower() or 'magazine' in m.lower() else 0.0
                            for m in mat_arr]).reshape(-1, 1)
        is_cardboard = np.array([1.0 if 'cardboard' in m.lower() or 'carton' in m.lower() else 0.0
                                 for m in mat_arr]).reshape(-1, 1)
        is_paper = np.array([1.0 if 'paper' in m.lower() else 0.0
                             for m in mat_arr]).reshape(-1, 1)
        feats.extend([is_plastic, is_book, is_cardboard, is_paper])

    return np.hstack(feats)


def get_feature_names(model_names, use_hierarchical=True):
    names = [f'pred_{m}' for m in model_names]
    names.append('mat_enc')
    names.extend(['pred_mean', 'pred_std', 'pred_max', 'pred_min', 'pred_range'])
    names.extend(['rank_std', 'rank_range'])
    names.extend([f'interact_{m}_dr' for m in model_names])
    if use_hierarchical:
        names.extend(['is_plastic', 'is_book', 'is_cardboard', 'is_paper'])
    return names


# ============================================================
# 4. 核心: val 训练 → test 应用 (LR)
# ============================================================
def val_train_test_apply(model_names, val_preds_dict, val_gt, val_mat,
                         test_preds_dict, test_gt, test_mat,
                         dr_map, le, C=0.3, use_hierarchical=True,
                         no_gt=False):
    val_common = sorted(
        set.intersection(*[set(val_preds_dict[m].keys()) for m in model_names])
        & set(val_gt.keys())
    )
    X_val_raw = np.array([[val_preds_dict[m][c] for m in model_names] for c in val_common])
    y_val = np.array([val_gt[c] for c in val_common])
    mat_val = np.array([val_mat.get(c, '') for c in val_common])

    if no_gt:
        test_common = sorted(
            set.intersection(*[set(test_preds_dict[m].keys()) for m in model_names])
        )
    else:
        test_common = sorted(
            set.intersection(*[set(test_preds_dict[m].keys()) for m in model_names])
            & set(test_gt.keys())
        )
    X_test_raw = np.array([[test_preds_dict[m][c] for m in model_names] for c in test_common])
    mat_test = np.array([test_mat.get(c, '') for c in test_common])

    X_val_feat = build_features(X_val_raw, mat_val, le, dr_map, model_names, use_hierarchical)
    X_test_feat = build_features(X_test_raw, mat_test, le, dr_map, model_names, use_hierarchical)

    clf = LogisticRegression(C=C, max_iter=1000, solver='lbfgs')
    clf.fit(X_val_feat, y_val)
    val_pred = clf.predict_proba(X_val_feat)[:, 1]
    val_ap = average_precision_score(y_val, val_pred)

    test_fusion_pred = clf.predict_proba(X_test_feat)[:, 1]

    test_fusion_ap = None
    test_raw_aps = {}
    if not no_gt:
        y_test = np.array([test_gt[c] for c in test_common])
        test_fusion_ap = average_precision_score(y_test, test_fusion_pred)
        for i, m in enumerate(model_names):
            test_raw_aps[m] = average_precision_score(y_test, X_test_raw[:, i])

    coefs = clf.coef_[0][:len(model_names)]
    coefs_full = clf.coef_[0]
    intercept = float(clf.intercept_[0])
    feature_names = get_feature_names(model_names, use_hierarchical)
    assert len(coefs_full) == len(feature_names), \
        f"系数维度 {len(coefs_full)} != 特征名维度 {len(feature_names)}"

    return {
        'val_ap': val_ap,
        'test_fusion_ap': test_fusion_ap,
        'test_raw_aps': test_raw_aps,
        'coefs': coefs,
        'coefs_full': coefs_full,
        'intercept': intercept,
        'feature_names': feature_names,
        'test_cids': test_common,
        'test_preds': test_fusion_pred,
        'val_n': len(val_common),
        'test_n': len(test_common),
    }


# ============================================================
# 5. val 上 K-fold CV 搜索最佳 C
# ============================================================
def c_search_cv(model_names, val_preds_dict, val_gt, val_mat,
                dr_map, le, use_hierarchical=True,
                c_candidates=None, n_folds=5, n_seeds=3):
    if c_candidates is None:
        c_candidates = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]

    val_common = sorted(
        set.intersection(*[set(val_preds_dict[m].keys()) for m in model_names])
        & set(val_gt.keys())
    )
    X_val_raw = np.array([[val_preds_dict[m][c] for m in model_names] for c in val_common])
    y_val = np.array([val_gt[c] for c in val_common])
    mat_val = np.array([val_mat.get(c, '') for c in val_common])

    X_val_feat = build_features(X_val_raw, mat_val, le, dr_map, model_names, use_hierarchical)

    results = []
    for C in c_candidates:
        fold_aps = []
        for seed in range(n_seeds):
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed * 111)
            oof = np.zeros(len(y_val))
            for tr, va in skf.split(X_val_feat, y_val):
                lr = LogisticRegression(C=C, max_iter=1000, solver='lbfgs')
                lr.fit(X_val_feat[tr], y_val[tr])
                oof[va] = lr.predict_proba(X_val_feat[va])[:, 1]
            fold_aps.append(average_precision_score(y_val, oof))
        mean_ap = np.mean(fold_aps)
        std_ap = np.std(fold_aps)
        results.append({'C': C, 'cv_ap_mean': mean_ap, 'cv_ap_std': std_ap})

    best = max(results, key=lambda x: x['cv_ap_mean'])
    return best, results


# ============================================================
# 6. LOO 贡献度分析
# ============================================================
def loo_analysis(model_names, val_preds_dict, val_gt, val_mat,
                 test_preds_dict, test_gt, test_mat,
                 dr_map, le, C=0.3, use_hierarchical=True):
    full = val_train_test_apply(
        model_names, val_preds_dict, val_gt, val_mat,
        test_preds_dict, test_gt, test_mat,
        dr_map, le, C, use_hierarchical=use_hierarchical)

    results = []
    for name in model_names:
        reduced = [m for m in model_names if m != name]
        red = val_train_test_apply(
            reduced, val_preds_dict, val_gt, val_mat,
            test_preds_dict, test_gt, test_mat,
            dr_map, le, C, use_hierarchical=use_hierarchical)
        results.append({
            'name': name,
            'test_ap_without': red['test_fusion_ap'],
            'delta': full['test_fusion_ap'] - red['test_fusion_ap'],
        })

    results.sort(key=lambda x: -x['delta'])
    return full, results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='多模型融合 (context-aware / LR)')
    parser.add_argument('-c', '--config', type=str, default='configs/ensemble.yaml',
                        help='配置文件路径 (default: configs/ensemble.yaml)')
    parser.add_argument('-s', '--save', type=str, default=None,
                        help='保存 test 预测到 CSV 文件路径')
    parser.add_argument('--C', type=float, default=0.05, help='LR 正则化参数 (default=0.05)')
    parser.add_argument('--c_search', action='store_true', help='LR 用 val CV 搜索最佳 C')
    parser.add_argument('--no_loo', action='store_true', help='跳过 LOO 贡献度分析')
    parser.add_argument('--no_hierarchical', action='store_true', help='去掉材质二值特征')
    parser.add_argument('--no_gt', action='store_true', help='测试集无 GT，跳过 AP 计算')
    parser.add_argument('--save_coefs', type=str, default=None,
                        help='LR 系数保存路径 (JSON)')
    parser.add_argument('--load_coefs', type=str, default=None,
                        help='加载已保存的 LR 系数进行纯推理 (JSON), 跳过训练')
    args = parser.parse_args()

    # 加载配置
    model_registry, method = load_config(args.config)
    model_names = list(model_registry.keys())

    use_hier = not args.no_hierarchical

    # 验证模型都有 val 和 test (推理模式下跳过 val 检查)
    if not args.load_coefs:
        no_val = [m for m in model_names if not model_registry.get(m, {}).get('val')]
        if no_val:
            print(f"ERROR: 以下模型无验证集路径: {no_val}")
            print(f"请在配置文件 {args.config} 中补充 val 路径")
            return
    no_test = [m for m in model_names if not model_registry.get(m, {}).get('test')]
    if no_test:
        print(f"ERROR: 以下模型无测试集路径: {no_test}")
        print(f"请在配置文件 {args.config} 中补充 test 路径")
        return

    dr_map = DEFECT_RATES

    # 加载 val 和 test 数据
    val_preds = {}
    val_gt, val_mat = {}, {}
    test_preds = {}
    test_gt, test_mat = {}, {}

    for name in model_names:
        info = model_registry[name]

        if not args.load_coefs:
            vgt, vmat = load_gt_and_mat(info['val'])
            vp = load_preds(info['val'])
            val_preds[name] = vp
            val_gt.update(vgt)
            val_mat.update(vmat)

        tp, tmat = load_preds_with_mat(info['test'])
        test_preds[name] = tp
        test_mat.update(tmat)
        if not args.no_gt:
            tgt, _ = load_gt_and_mat(info['test'])
            test_gt.update(tgt)

    # 数据完整性
    if args.load_coefs:
        val_cids = []
    else:
        val_cids = sorted(set.intersection(*[set(val_preds[m].keys()) for m in model_names]) & set(val_gt.keys()))
    if args.no_gt:
        test_cids = sorted(set.intersection(*[set(test_preds[m].keys()) for m in model_names]))
    else:
        test_cids = sorted(set.intersection(*[set(test_preds[m].keys()) for m in model_names]) & set(test_gt.keys()))

    print(f"配置文件: {args.config}")
    print(f"方法: {method} (context-aware)")
    print(f"模型数: {len(model_names)}")
    print(f"模型列表: {model_names}")
    if not args.load_coefs:
        print(f"Val 样本数: {len(val_cids)}")
    else:
        print(f"模式: --load_coefs (纯推理)")
    print(f"Test 样本数: {len(test_cids)}")
    if args.no_gt:
        print(f"模式: --no_gt (测试集无 GT，仅输出预测)")
    print()

    # 缺失检查 (仅训练模式)
    if not args.load_coefs:
        for name in model_names:
            v_n = len(set(val_preds[name].keys()) & set(val_gt.keys()))
            t_n = len(set(test_preds[name].keys()))
            v_miss = len(val_cids) - len(set(val_preds[name].keys()) & set(val_cids))
            t_miss = len(test_cids) - len(set(test_preds[name].keys()) & set(test_cids))
            issues = []
            if v_miss > 0:
                issues.append(f'val缺{v_miss}')
            if t_miss > 0:
                issues.append(f'test缺{t_miss}')
            status = f'  ⚠ {", ".join(issues)}' if issues else '  ✓'
            print(f"  {name:20s}: val={v_n}, test={t_n}  {status}")

    # LabelEncoder
    all_materials = set(val_mat.values()) | set(test_mat.values())
    le = LabelEncoder()
    le.fit(list(all_materials))

    # ============================================================
    # 0. --load_coefs 纯推理模式
    # ============================================================
    if args.load_coefs:
        with open(args.load_coefs) as f:
            coefs_data = json.load(f)

        saved_model_names = coefs_data['model_names']
        saved_coefs = np.array(coefs_data['coefficients'])
        saved_intercept = coefs_data['intercept']
        saved_feature_names = coefs_data['feature_names']
        saved_use_hier = coefs_data.get('use_hierarchical', True)

        dr_map = coefs_data.get('defect_rates', dr_map)

        if 'material_classes' in coefs_data:
            le = LabelEncoder()
            le.fit(coefs_data['material_classes'])

        assert len(saved_coefs) == len(saved_feature_names), \
            f"系数维度 {len(saved_coefs)} != 特征名维度 {len(saved_feature_names)}"

        for m in saved_model_names:
            if m not in test_preds and m in model_registry:
                info = model_registry[m]
                tp, tmat = load_preds_with_mat(info['test'])
                test_preds[m] = tp
                test_mat.update(tmat)
                if not args.no_gt:
                    tgt, _ = load_gt_and_mat(info['test'])
                    test_gt.update(tgt)
            elif m not in test_preds and m not in model_registry:
                print(f"⚠ 模型 {m} 不在配置文件中，无法加载 test 数据")

        effective_models = [m for m in saved_model_names if m in test_preds]
        if len(effective_models) != len(saved_model_names):
            missing = [m for m in saved_model_names if m not in test_preds]
            print(f"⚠ 以下模型缺少 test 数据: {missing}")

        test_common = sorted(
            set.intersection(*[set(test_preds[m].keys()) for m in effective_models])
        )
        if not args.no_gt:
            test_common = sorted(set(test_common) & set(test_gt.keys()))

        X_test_raw = np.array([[test_preds[m][c] for m in effective_models] for c in test_common])
        mat_test = np.array([test_mat.get(c, '') for c in test_common])

        X_test_feat = build_features(X_test_raw, mat_test, le, dr_map, effective_models, saved_use_hier)

        z = X_test_feat @ saved_coefs + saved_intercept
        test_fusion_pred = 1.0 / (1.0 + np.exp(-z))

        print(f"\n{'=' * 80}")
        print(f"纯推理模式 (加载权重: {args.load_coefs})")
        print("=" * 80)
        print(f"  模型列表     = {effective_models}")
        print(f"  特征维度     = {len(saved_feature_names)}")
        print(f"  截距         = {saved_intercept:+.6f}")
        print(f"  Test 样本    = {len(test_common)}")

        print(f"\n  LR 全部特征系数 (按绝对值排序):")
        print(f"    {'intercept':30s}: {saved_intercept:+.6f}")
        for fname, coef in sorted(zip(saved_feature_names, saved_coefs), key=lambda x: -abs(x[1])):
            print(f"    {fname:30s}: {coef:+.6f}")

        test_fusion_ap = None
        if not args.no_gt and test_gt:
            y_test = np.array([test_gt[c] for c in test_common])
            test_fusion_ap = average_precision_score(y_test, test_fusion_pred)
            print(f"\n  Test AP = {test_fusion_ap:.4f}")

        if args.save:
            with open(args.save, 'w', newline='') as f:
                writer = csv_mod.writer(f)
                writer.writerow(['capture_id', 'pred'])
                for cid, pred in zip(test_common, test_fusion_pred):
                    writer.writerow([cid, float(pred)])
            print(f"\n  已保存到 {args.save} ({len(test_common)} 条)")

        print("\n===== Done (推理模式) =====")
        return

    # ============================================================
    # 1. 单模型 Test AP
    # ============================================================
    single_aps = {}
    if not args.no_gt:
        print("\n" + "=" * 80)
        print("单模型 Test AP (原始预测)")
        print("=" * 80)
        for name in model_names:
            pred_arr = np.array([test_preds[name][c] for c in test_cids])
            y_arr = np.array([test_gt[c] for c in test_cids])
            ap = average_precision_score(y_arr, pred_arr)
            single_aps[name] = ap
            print(f"  {name:20s}: AP={ap:.4f}")

    # ============================================================
    # 2. LR C 搜索 (可选)
    # ============================================================
    lr_C = args.C
    if args.c_search:
        print("\n" + "=" * 80)
        print("LR C 参数搜索 (val 5-fold CV, 3 seeds)")
        print("=" * 80)
        best, all_results = c_search_cv(
            model_names, val_preds, val_gt, val_mat,
            dr_map, le, use_hierarchical=use_hier)
        for r in all_results:
            marker = ' ←' if r['C'] == best['C'] else ''
            print(f"  C={r['C']:<5}: CV AP={r['cv_ap_mean']:.4f} ± {r['cv_ap_std']:.4f}{marker}")
        print(f"\n  最佳 C={best['C']} (CV AP={best['cv_ap_mean']:.4f} ± {best['cv_ap_std']:.4f})")
        lr_C = best['C']

    # ============================================================
    # 3a. dr_adjusted_mean 融合 (无训练)
    # ============================================================
    if method == 'dr_adjusted_mean':
        print("\n" + "=" * 80)
        print("DR Adjusted Mean 融合 (无训练)")
        print("=" * 80)

        X_val_raw = np.array([[val_preds[m][c] for m in model_names] for c in val_cids])
        y_val = np.array([val_gt[c] for c in val_cids])
        mat_val_arr = np.array([val_mat.get(c, '') for c in val_cids])

        X_test_raw = np.array([[test_preds[m][c] for m in model_names] for c in test_cids])
        mat_test_arr = np.array([test_mat.get(c, '') for c in test_cids])

        val_fused = fuse_dr_adjusted_mean(X_val_raw, mat_val_arr, dr_map)
        val_ap = average_precision_score(y_val, val_fused)

        test_fused = fuse_dr_adjusted_mean(X_test_raw, mat_test_arr, dr_map)

        test_ap = None
        test_raw_aps = {}
        if not args.no_gt:
            y_test = np.array([test_gt[c] for c in test_cids])
            test_ap = average_precision_score(y_test, test_fused)
            for i, m in enumerate(model_names):
                test_raw_aps[m] = average_precision_score(y_test, X_test_raw[:, i])

        print(f"  Val  AP   = {val_ap:.4f}")
        if not args.no_gt:
            print(f"  Test AP   = {test_ap:.4f}")
        print(f"  Val 样本  = {len(val_cids)}")
        print(f"  Test 样本 = {len(test_cids)}")

        if not args.no_gt:
            best_single = max(single_aps.values())
            best_single_name = max(single_aps, key=single_aps.get)
            print(f"  最佳单模型 = {best_single:.4f} ({best_single_name})")
            print(f"  融合提升   = {test_ap - best_single:+.4f}")

            print(f"\n  单模型 vs 融合对比:")
            for name in model_names:
                raw_ap = test_raw_aps[name]
                delta = test_ap - raw_ap
                print(f"    {name:20s}: {raw_ap:.4f} → 融合 {test_ap:.4f} ({delta:+.4f})")

        # LOO
        if not args.no_loo and not args.no_gt and len(model_names) > 1:
            print("\n" + "=" * 80)
            print(f"LOO 贡献度分析 (dr_adjusted_mean)")
            print("=" * 80)

            y_test = np.array([test_gt[c] for c in test_cids])
            loo_results = []
            for name in model_names:
                reduced = [m for m in model_names if m != name]
                X_test_red = np.array([[test_preds[m][c] for m in reduced] for c in test_cids])
                test_red = fuse_dr_adjusted_mean(X_test_red, mat_test_arr, dr_map)
                red_ap = average_precision_score(y_test, test_red)
                loo_results.append({
                    'name': name,
                    'test_ap_without': red_ap,
                    'delta': test_ap - red_ap,
                })
            loo_results.sort(key=lambda x: -x['delta'])

            print(f"  {'排名':>4s}  {'模型':20s}  {'Test AP(去)':>12s}  {'LOO Δ':>10s}  {'角色':>6s}")
            print("  " + "-" * 60)
            for rank, r in enumerate(loo_results, 1):
                role = '核心' if r['delta'] > 0.005 else ('有用' if r['delta'] > 0 else '冗余')
                print(f"  {rank:4d}  {r['name']:20s}  {r['test_ap_without']:12.4f}  {r['delta']:+10.6f}  {role:>6s}")

        # 保存 CSV
        if args.save:
            with open(args.save, 'w', newline='') as f:
                writer = csv_mod.writer(f)
                writer.writerow(['capture_id', 'pred'])
                for cid, pred in zip(test_cids, test_fused):
                    writer.writerow([cid, float(pred)])
            print(f"\n  已保存到 {args.save} ({len(test_cids)} 条)")

        print("\n===== Done =====")
        return

    # ============================================================
    # 3b. LR 融合 (context-aware)
    # ============================================================
    print("\n" + "=" * 80)
    print(f"Logistic Regression 融合 (context-aware, C={lr_C})")
    print("=" * 80)

    res = val_train_test_apply(
        model_names, val_preds, val_gt, val_mat,
        test_preds, test_gt, test_mat,
        dr_map, le, C=lr_C, use_hierarchical=use_hier, no_gt=args.no_gt)

    n_feat = len(model_names) + 1 + 5 + 2 + len(model_names) + (4 if use_hier else 0)
    print(f"  Val  AP   = {res['val_ap']:.4f}  (训练集拟合)")
    if not args.no_gt:
        print(f"  Test AP   = {res['test_fusion_ap']:.4f}  (无泄露)")
    print(f"  Val 样本  = {res['val_n']}")
    print(f"  Test 样本 = {res['test_n']}")
    print(f"  特征维度  = {n_feat}")

    if not args.no_gt:
        best_single = max(single_aps.values())
        best_single_name = max(single_aps, key=single_aps.get)
        print(f"  最佳单模型 = {best_single:.4f} ({best_single_name})")
        print(f"  融合提升   = {res['test_fusion_ap'] - best_single:+.4f}")

        print(f"\n  单模型 vs 融合对比:")
        for name in model_names:
            raw_ap = res['test_raw_aps'][name]
            delta = res['test_fusion_ap'] - raw_ap
            print(f"    {name:20s}: {raw_ap:.4f} → 融合 {res['test_fusion_ap']:.4f} ({delta:+.4f})")

    if res['coefs'] is not None:
        print(f"\n  LR 原始预测系数 (按绝对值排序):")
        for name, coef in sorted(zip(model_names, res['coefs']), key=lambda x: -abs(x[1])):
            print(f"    {name:20s}: {coef:+.4f}")

        print(f"\n  LR 全部特征系数 (按绝对值排序):")
        fnames = res['feature_names']
        cfull = res['coefs_full']
        intercept = res['intercept']
        print(f"    {'intercept':30s}: {intercept:+.6f}")
        for fname, coef in sorted(zip(fnames, cfull), key=lambda x: -abs(x[1])):
            print(f"    {fname:30s}: {coef:+.6f}")

    # ============================================================
    # 4. LOO 贡献度分析
    # ============================================================
    if not args.no_loo and not args.no_gt and len(model_names) > 1:
        print("\n" + "=" * 80)
        print(f"LOO 贡献度分析 (LR, val 训练, test 评估)")
        print("=" * 80)

        full, loo_results = loo_analysis(
            model_names, val_preds, val_gt, val_mat,
            test_preds, test_gt, test_mat,
            dr_map, le, C=lr_C, use_hierarchical=use_hier)

        print(f"  {'排名':>4s}  {'模型':20s}  {'Test AP(去)':>12s}  {'LOO Δ':>10s}  {'角色':>6s}")
        print("  " + "-" * 60)
        for rank, r in enumerate(loo_results, 1):
            name = r['name']
            role = '核心' if r['delta'] > 0.005 else ('有用' if r['delta'] > 0 else '冗余')
            print(f"  {rank:4d}  {name:20s}  {r['test_ap_without']:12.4f}  {r['delta']:+10.6f}  {role:>6s}")

    # ============================================================
    # 5. 输出 CSV
    # ============================================================
    if args.save:
        with open(args.save, 'w', newline='') as f:
            writer = csv_mod.writer(f)
            writer.writerow(['capture_id', 'pred'])
            for cid, pred in zip(res['test_cids'], res['test_preds']):
                writer.writerow([cid, float(pred)])
        print(f"\n  已保存到 {args.save} ({len(res['test_cids'])} 条)")

    # ============================================================
    # 6. 保存 LR 系数
    # ============================================================
    if args.save_coefs and res.get('coefs_full') is not None:
        coefs_data = {
            'method': 'lr',
            'C': lr_C,
            'model_names': model_names,
            'feature_names': res['feature_names'],
            'coefficients': [float(c) for c in res['coefs_full']],
            'intercept': res['intercept'],
            'use_hierarchical': use_hier,
            'defect_rates': dr_map,
            'material_classes': list(le.classes_),
        }
        with open(args.save_coefs, 'w') as f:
            json.dump(coefs_data, f, indent=2, ensure_ascii=False)
        print(f"\n  LR 系数已保存到 {args.save_coefs}")

    print("\n===== Done =====")


if __name__ == '__main__':
    main()