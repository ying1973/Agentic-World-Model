# 实现FVD、LPIPS、SSIM, PSNR、DINOv2、MSE、CLIP(image, text)、CLIP Image-Image Similarity、CLIP-based similarity、Cosine Similarity、
# Exact Match、BLEU、BERTScore、Pass@k、Execution Rate、等指标

from bert_score import score


def fvd():
    pass

def bert_score(pred: str, ref: str, model_type: str = "microsoft/deberta-xlarge-mnli"):
    """
    计算 BERTScore.
    
    Args:
        pred (str): 生成文本（candidate / hypothesis）
        ref (str): 参考文本（reference）
        model_type (str): 使用的预训练模型，默认 "microsoft/deberta-xlarge-mnli"
    
    Returns:
        dict: 包含 precision, recall, f1-score 三个值
    """
    # bert_score.score 要求 list of strings
    cands = [pred]
    refs = [ref]

    P, R, F1 = score(
        cands,
        refs,
        model_type=model_type,
        num_layers=40,
        lang="en",
        verbose=False,
    )

    return {
        "precision": float(P[0]),
        "recall": float(R[0]),
        "f1": float(F1[0])
    }

if __name__ == "__main__":
    pred = "The cat is on the mat."
    ref = "China is a country."
    result = bert_score(pred, ref)
    print(f"BERTScore: Precision={result['precision']}, Recall={result['recall']}, F1={result['f1']}")