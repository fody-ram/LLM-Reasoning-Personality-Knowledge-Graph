 {r:<8.2f} | {f:<8.2f}")
    # Relationships (works_at)
    true_rels = [rel for rel in ground_truth["relationships"] if rel[0] == person]
    pred_rels = [tuple(rel) for rel in predicted_relationships if rel[0] == person]
    p, r, f = evaluate_extraction(pred_rels, true_rels)
    print(f"{person:<10} | {'Works_at':<12} | {p:<8.2f} | {r:<8.2f} | {f:<8.2f}")
