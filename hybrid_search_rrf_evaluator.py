import pandas as pd
import numpy as np
import time
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# -------------------------------------------------
# 1. 데이터 로드 및 전처리
# -------------------------------------------------
df = pd.read_csv('data/dataset.csv')
df = df.drop_duplicates(subset=['SEQ'], keep='last').reset_index(drop=True)

df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')
df['RESTRNT_AM'] = pd.to_numeric(df['RESTRNT_AM'], errors='coerce')
df['summary'] = df['summary'].fillna("")

print(f"전체 데이터 수: {len(df)}")

# -------------------------------------------------
# 2. 임베딩 모델 로드 및 캐싱 (Dense)
# -------------------------------------------------
print("한국어 임베딩 모델 로드 중...")
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

EMBEDDING_FILE = 'data/corpus_embeddings.npy'
if os.path.exists(EMBEDDING_FILE):
    corpus_embeddings = np.load(EMBEDDING_FILE)
else:
    corpus_embeddings = model.encode(df['summary'].tolist(), show_progress_bar=True)
    np.save(EMBEDDING_FILE, corpus_embeddings)

# -------------------------------------------------
# 3. BM25 인덱스 생성 (Sparse)
# -------------------------------------------------
print("BM25 인덱스 생성 중...")
def preprocess_text(text):
    text = str(text)
    text = re.sub(r"[^가-힣0-9a-zA-Z\s]", "", text)
    return text.lower().split()

tokenized_corpus = [preprocess_text(doc) for doc in df['summary'].tolist()]
bm25 = BM25Okapi(tokenized_corpus)
print("준비 완료\n")

# -------------------------------------------------
# 유틸리티 함수
# -------------------------------------------------
def calculate_rrf(k_rank, v_rank, k=60):
    score = 0
    if not np.isnan(k_rank):
        score += 1 / (k + k_rank)
    if not np.isnan(v_rank):
        score += 1 / (k + v_rank)
    return score

# Recall@K 및 Hit@K 계산을 위한 Ground Truth(정답) 생성기
# 정의: 쿼리 조건(지역/나이)을 만족하면서 요식업 지출이 상위 10%인 고객
def get_ground_truth(df, query_sido, query_age_min, query_age_max):
    condition_matched = df[
        (df['HOUS_SIDO_NM'] == query_sido) &
        (df['AGE'] >= query_age_min) &
        (df['AGE'] <= query_age_max)
    ]
    if len(condition_matched) == 0:
        return []
    threshold = condition_matched['RESTRNT_AM'].quantile(0.9)
    relevant_seqs = condition_matched[condition_matched['RESTRNT_AM'] >= threshold]['SEQ'].tolist()
    return relevant_seqs

def calc_recall_hit_at_k(pred_seqs, true_seqs, k=10):
    if len(true_seqs) == 0:
        return 0.0, 0
    pred_k = set(pred_seqs[:k])
    true_set = set(true_seqs)
    hits = len(pred_k & true_set)
    recall = hits / len(true_set)
    hit_at_k = 1 if hits > 0 else 0
    return recall, hit_at_k

# -------------------------------------------------
# 4종 비교 실험 함수
# -------------------------------------------------
def run_rrf_experiment(query_text, query_sido='서울', query_age_min=30, query_age_max=39, k=10):
    print("="*80)
    print(f"* 실험 쿼리: {query_text} (조건: {query_sido}, {query_age_min}대)")
    print("="*80)

    # 평가를 위한 정답(Ground Truth) 셋 구축
    gt_seqs = get_ground_truth(df, query_sido, query_age_min, query_age_max)
    
    # -------------------------------------------------
    # Retriever 1: Dense 단독
    # -------------------------------------------------
    start_dense = time.time()
    query_embedding = model.encode([query_text])
    df['similarity_score'] = cosine_similarity(query_embedding, corpus_embeddings)[0]
    dense_result = df.sort_values(by='similarity_score', ascending=False).head(1000).copy().reset_index(drop=True)
    dense_result['vector_rank'] = dense_result.index + 1
    end_dense = time.time()

    # -------------------------------------------------
    # Retriever 2: BM25 단독
    # -------------------------------------------------
    start_bm25 = time.time()
    bm25_query_str = f"{query_sido} {query_age_min}대 요식업" 
    df['bm25_score'] = bm25.get_scores(preprocess_text(bm25_query_str))
    bm25_result = df[df['bm25_score'] > 0].sort_values(by=['bm25_score', 'RESTRNT_AM'], ascending=[False, False]).head(1000).copy().reset_index(drop=True)
    bm25_result['keyword_rank'] = bm25_result.index + 1
    end_bm25 = time.time()

    # -------------------------------------------------
    # SQL 필터 통과자만 남기기 (변인 통제)
    # -------------------------------------------------
    start_filter = time.time()
    valid_candidates = df[
        (df['HOUS_SIDO_NM'] == query_sido) &
        (df['AGE'] >= query_age_min) &
        (df['AGE'] <= query_age_max)
    ]['SEQ']
    
    filtered_dense = dense_result[dense_result['SEQ'].isin(valid_candidates)].copy().reset_index(drop=True)
    filtered_bm25 = bm25_result[bm25_result['SEQ'].isin(valid_candidates)].copy().reset_index(drop=True)
    end_filter = time.time()

    # -------------------------------------------------
    # Retriever 3: Score Fusion (단순 가중합)
    # -------------------------------------------------
    start_score = time.time()
    max_bm25, min_bm25 = filtered_bm25['bm25_score'].max(), filtered_bm25['bm25_score'].min()
    filtered_bm25['bm25_norm'] = (filtered_bm25['bm25_score'] - min_bm25) / (max_bm25 - min_bm25) if max_bm25 != min_bm25 else 0

    max_dense, min_dense = filtered_dense['similarity_score'].max(), filtered_dense['similarity_score'].min()
    filtered_dense['dense_norm'] = (filtered_dense['similarity_score'] - min_dense) / (max_dense - min_dense) if max_dense != min_dense else 0

    fusion_base = pd.merge(
        filtered_bm25[['SEQ', 'bm25_score', 'bm25_norm', 'keyword_rank']],
        filtered_dense[['SEQ', 'similarity_score', 'dense_norm', 'vector_rank']],
        on='SEQ', how='inner'
    )
    
    fusion_base['weighted_score'] = (0.5 * fusion_base['bm25_norm']) + (0.5 * fusion_base['dense_norm'])
    score_fusion_result = fusion_base.merge(df.drop(columns=['bm25_score', 'similarity_score'], errors='ignore'), on='SEQ').sort_values(by='weighted_score', ascending=False)
    end_score = time.time()

    # -------------------------------------------------
    # Retriever 4: Rank Fusion (RRF)
    # -------------------------------------------------
    start_rrf = time.time()
    fusion_base['rrf_score'] = fusion_base.apply(lambda x: calculate_rrf(x['keyword_rank'], x['vector_rank']), axis=1)
    rrf_fusion_result = fusion_base.merge(df.drop(columns=['bm25_score', 'similarity_score'], errors='ignore'), on='SEQ').sort_values(by='rrf_score', ascending=False)
    end_rrf = time.time()

    # =================================================
    # 지표 계산 및 출력
    # =================================================
    def calc_accuracy(results_df):
        if len(results_df) == 0: return 0
        correct = results_df[
            (results_df['HOUS_SIDO_NM'] == query_sido) &
            (results_df['AGE'] >= query_age_min) &
            (results_df['AGE'] <= query_age_max)
        ]
        return (len(correct) / len(results_df) * 100)

    print("\n[1. Accuracy 비교 (필터 전/후 조건 일치율)]")
    print(f" - 1. Dense (필터 전): {calc_accuracy(dense_result.head(10)):.1f}%")
    print(f" - 2. BM25 (필터 전):  {calc_accuracy(bm25_result.head(10)):.1f}%")
    print(f" - 3. Score Fusion (필터 후): {calc_accuracy(score_fusion_result.head(10)):.1f}%")
    print(f" - 4. Rank Fusion (필터 후):  {calc_accuracy(rrf_fusion_result.head(10)):.1f}%")

    print("\n[2. Latency 비교 (ms)]")
    print(f" - Dense 검색: {(end_dense - start_dense)*1000:.2f} ms")
    print(f" - BM25 검색:  {(end_bm25 - start_bm25)*1000:.2f} ms")
    print(f" - SQL 필터링: {(end_filter - start_filter)*1000:.2f} ms")
    print(f" - Score Fusion 연산: {(end_score - start_score)*1000:.2f} ms")
    print(f" - Rank Fusion (RRF) 연산: {(end_rrf - start_rrf)*1000:.2f} ms")

    print("\n[3. 연산량 및 모수 축소 정보]")
    print(f" - 초기 전체 모수: {len(df):,}")
    print(f" - SQL 필터 통과자(Valid): {len(valid_candidates):,}")
    print(f" - 불필요한 연산 축소율: {(1 - len(valid_candidates)/len(df))*100:.1f}%")

    print(f"\n[4. 검색 성능 평가 (Recall@{k}, Hit@{k})]")
    print(f" * Ground Truth(정답): {query_sido} {query_age_min}대 중 요식업 지출 상위 10% 고객 ({len(gt_seqs)}명)")
    
    metrics = {
        "1. Dense + SQL": filtered_dense['SEQ'].tolist(),
        "2. BM25 + SQL": filtered_bm25['SEQ'].tolist(),
        "3. Score Fusion": score_fusion_result['SEQ'].tolist(),
        "4. Rank Fusion": rrf_fusion_result['SEQ'].tolist()
    }
    
    for name, seqs in metrics.items():
        recall, hit = calc_recall_hit_at_k(seqs, gt_seqs, k=k)
        print(f" - {name:15} | Recall@{k}: {recall:.4f} | Hit@{k}: {hit}")

    print("\n[5. 최종 랭킹 품질 (Top 3 출력)]")
    cols = ['SEQ', 'HOUS_SIDO_NM', 'AGE', 'RESTRNT_AM']

    print("\n1. Dense + SQL (문제: 문맥만 보고 실제 지출액을 무시함)")
    print(filtered_dense[cols + ['similarity_score']].head(3).to_string(index=False))

    print("\n2. BM25 + SQL (문제: 단어만 매칭되어 지출액이 0원인 깡통 데이터 1위)")
    print(filtered_bm25[cols + ['bm25_score']].head(3).to_string(index=False))

    print("\n3. Score Fusion (문제: 정규화의 한계로 특정 모델에 랭킹이 휘둘림)")
    print(score_fusion_result[cols + ['bm25_norm', 'dense_norm', 'weighted_score']].head(3).to_string(index=False))

    print("\n4. Rank Fusion (RRF) (해결: 안정적인 융합으로 진짜 VIP 고객 추출)")
    if len(rrf_fusion_result) > 0:
        print(rrf_fusion_result[cols + ['keyword_rank', 'vector_rank', 'rrf_score']].head(3).to_string(index=False))
    else:
        print("교집합 결과 없음")
    print("="*80)

# 실행 (K=10 기준)
run_rrf_experiment("외식 소비가 많은 고객", query_sido='서울', query_age_min=30, query_age_max=39, k=10)