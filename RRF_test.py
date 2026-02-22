import pandas as pd
import numpy as np
import time
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
# 2. 임베딩 모델 로드
# -------------------------------------------------
print("한국어 임베딩 모델 준비 중...")
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# -------------------------------------------------
# 3. 코퍼스 임베딩 캐싱
# -------------------------------------------------
EMBEDDING_FILE = 'data/corpus_embeddings.npy'

if os.path.exists(EMBEDDING_FILE):
    print("저장된 벡터 로드 중...")
    corpus_embeddings = np.load(EMBEDDING_FILE)
else:
    print("전체 벡터 생성 중 (최초 1회)...")
    corpus_embeddings = model.encode(df['summary'].tolist(), show_progress_bar=True)
    np.save(EMBEDDING_FILE, corpus_embeddings)
    print("벡터 저장 완료")

print("준비 완료\n")


# -------------------------------------------------
# RRF 계산 함수
# -------------------------------------------------
def calculate_rrf(k_rank, v_rank, k=60):
    score = 0
    if not np.isnan(k_rank):
        score += 1 / (k + k_rank)
    if not np.isnan(v_rank):
        score += 1 / (k + v_rank)
    return score


# -------------------------------------------------
# RRF 비교 실험 함수
# -------------------------------------------------
def run_rrf_experiment(query_text, query_sido='서울', query_age_min=30, query_age_max=39):

    print("="*70)
    print(f"실험 쿼리: {query_text}")
    print("="*70)

    start_total = time.time()

    # -------------------------------------------------
    # Dense 전체 검색 (Baseline)
    # -------------------------------------------------
    start_dense = time.time()

    query_embedding = model.encode([query_text])
    cos_scores = cosine_similarity(query_embedding, corpus_embeddings)[0]

    df['similarity_score'] = cos_scores
    dense_result = df.sort_values(by='similarity_score', ascending=False).head(100).copy().reset_index(drop=True)
    dense_result['vector_rank'] = dense_result.index + 1

    end_dense = time.time()

    # -------------------------------------------------
    # SQL 사전 필터링 (구조적 필터)
    # -------------------------------------------------
    start_filter = time.time()

    filtered_df = df[
        (df['HOUS_SIDO_NM'] == query_sido) &
        (df['AGE'] >= query_age_min) &
        (df['AGE'] <= query_age_max)
    ].copy()

    filtered_df = filtered_df.sort_values(by='RESTRNT_AM', ascending=False).reset_index(drop=True)
    filtered_df['keyword_rank'] = filtered_df.index + 1

    end_filter = time.time()

    # -------------------------------------------------
    # RRF 융합 (교집합)
    # -------------------------------------------------
    start_rrf = time.time()

    merged = pd.merge(
        filtered_df[['SEQ', 'keyword_rank']],
        dense_result[['SEQ', 'vector_rank']],
        on='SEQ',
        how='inner'
    )

    merged['rrf_score'] = merged.apply(
        lambda x: calculate_rrf(x['keyword_rank'], x['vector_rank']), axis=1
    )

    final_result = merged.merge(df, on='SEQ')
    final_result = final_result.sort_values(by='rrf_score', ascending=False)

    end_rrf = time.time()
    end_total = time.time()

    # -------------------------------------------------
    # 정확도 계산
    # -------------------------------------------------
    def calc_accuracy(results_df):
        correct = results_df[
            (results_df['HOUS_SIDO_NM'] == query_sido) &
            (results_df['AGE'] >= query_age_min) &
            (results_df['AGE'] <= query_age_max)
        ]
        return (len(correct) / len(results_df) * 100) if len(results_df) > 0 else 0

    dense_acc = calc_accuracy(dense_result.head(10))
    rrf_acc = calc_accuracy(final_result.head(10))

    # -------------------------------------------------
    # 결과 출력
    # -------------------------------------------------
    print("\n[1. Accuracy 비교]")
    print(f" - Dense 단독 Top10 정확도: {dense_acc:.1f}%")
    print(f" - RRF 하이브리드 Top10 정확도: {rrf_acc:.1f}%")

    print("\n[2. Latency 비교]")
    print(f" - Dense 전체 검색 시간: {(end_dense - start_dense)*1000:.2f} ms")
    print(f" - 사전 필터링 시간: {(end_filter - start_filter)*1000:.2f} ms")
    print(f" - RRF 연산 시간: {(end_rrf - start_rrf)*1000:.2f} ms")
    print(f" - 전체 수행 시간: {(end_total - start_total)*1000:.2f} ms")

    print("\n[3. 연산량 정보]")
    print(f" - 전체 벡터 비교 수: {len(df)}")
    print(f" - 필터 후 후보 수: {len(filtered_df)}")

    reduction = (1 - len(filtered_df)/len(df)) * 100
    print(f" - 후보 축소율: {reduction:.2f}%")

    print("\n[4. 랭킹 품질 비교]")

    cols = ['SEQ', 'HOUS_SIDO_NM', 'AGE', 'similarity_score', 'RESTRNT_AM']

    print("\nDense 단독 Top3")
    print(dense_result[cols + ['vector_rank']].head(3).to_string(index=False))

    print("\nRF 하이브리드 Top3")
    if len(final_result) > 0:
        print(final_result[cols + ['keyword_rank','vector_rank','rrf_score']].head(3).to_string(index=False))
    else:
        print("교집합 결과 없음")

    print("="*70)


# -------------------------------------------------
# 실험 실행
# -------------------------------------------------
# 핵심: 조건을 쿼리에서 제거해야 차이가 드러남
user_query = "외식 소비가 많은 고객"

run_rrf_experiment(
    query_text=user_query,
    query_sido='서울',
    query_age_min=30,
    query_age_max=39
)