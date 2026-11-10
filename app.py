@app.route("/match")
def match_items():
    import heapq

    global FINGERPRINT_CACHE
    FINGERPRINT_CACHE.clear()

    matches_heap = []
    item_limit = max(10, min(int(os.environ.get("MATCH_ITEM_LIMIT", "50")), 100))
    result_limit = max(10, min(int(os.environ.get("MATCH_RESULT_LIMIT", "30")), 50))

    try:
        lost_items = (
            LostItem.query
            .filter(LostItem.trang_thai != "DA_TIM")
            .order_by(LostItem.id.desc())
            .limit(item_limit)
            .all()
        )
        found_items = (
            FoundItem.query
            .filter(FoundItem.trang_thai != "DA_TRA")
            .order_by(FoundItem.id.desc())
            .limit(item_limit)
            .all()
        )
    except Exception as e:
        print(f"Lỗi tải dữ liệu đối sánh: {e}")
        lost_items, found_items = [], []

    for lost in lost_items:
        for found in found_items:
            try:
                key1 = get_core_keyword(lost.ten_do)
                key2 = get_core_keyword(found.ten_do)
                ten_sim = similarity(lost.ten_do, found.ten_do)
                is_same_key = key1 == key2 and bool(key1)

                # Bỏ qua các cặp hoàn toàn không liên quan nhau
                if not is_same_key and ten_sim < 0.35:
                    continue

                ten_score = 15 if is_same_key else int(ten_sim * 15)

                m1 = str(lost.mau_sac or "").strip().lower()
                m2 = str(found.mau_sac or "").strip().lower()
                mau_score = 0
                if m1 not in ("", "none", "null") and m2 not in ("", "none", "null"):
                    mau_score = int(similarity(m1, m2) * 15)
                elif is_same_key:
                    mau_score = 10

                v1 = str(lost.noi_mat or "").strip().lower()
                v2 = str(found.noi_nhat or "").strip().lower()
                vi_tri_score = 0
                if v1 and v2 and v1 not in ("none", "null") and v2 not in ("none", "null"):
                    vi_tri = similarity(v1, v2)
                    vi_tri_score = 10 if vi_tri > 0.8 else (5 if vi_tri > 0.5 else 0)

                ngay_score = date_score(lost.ngay_mat, found.ngay_nhat)
                text_total = ten_score + mau_score + vi_tri_score + ngay_score

                has1 = check_real_image(lost.hinh_anh)
                has2 = check_real_image(found.hinh_anh)
                
                # TÍNH ĐIỂM ÁNH CHUẨN XÁC
                if has1 and has2:
                    anh_score, image_ratio = image_score(
                        lost.hinh_anh, found.hinh_anh,
                        is_same_keyword=is_same_key
                    )
                    # Nếu cùng tên đồ vật mà tính ảnh thất bại do lỗi mạng, lấy mức sàn 25đ
                    if anh_score == 0 and is_same_key:
                        anh_score, image_ratio = 25, 0.5
                else:
                    anh_score, image_ratio = 0, 0.0

                score = anh_score + text_total

                if not is_same_key and ten_sim < 0.60:
                    score = min(score, 35)

                if score < 20:
                    continue

                if score >= 75:
                    ket_luan = "🟢 Khả năng trùng khớp cao"
                elif score >= 55:
                    ket_luan = "🟡 Có khả năng trùng khớp"
                elif score >= 38:
                    ket_luan = "🟠 Có một số điểm tương đồng"
                else:
                    ket_luan = "🔴 Cần kiểm tra thêm"

                reasons = []
                if anh_score >= 18:
                    reasons.append("Hình ảnh có điểm tương đồng")
                if ten_score >= 10:
                    reasons.append("Tên đồ tương đồng")
                if mau_score >= 10:
                    reasons.append("Màu sắc tương đồng")
                if vi_tri_score >= 5:
                    reasons.append("Vị trí liên quan")
                if ngay_score >= 6:
                    reasons.append("Thời gian khớp/gần nhau")

                image_confidence = int(min(100, image_ratio * 100))
                details = {
                    "hinh_anh": anh_score,
                    "ten_do": ten_score,
                    "mau_sac": mau_score,
                    "vi_tri": vi_tri_score,
                    "ngay": ngay_score,
                    "score_image": anh_score,
                    "score_name": ten_score,
                    "score_color": mau_score,
                    "score_location": vi_tri_score,
                    "score_date": ngay_score,
                    "image_confidence": image_confidence
                }

                result = {
                    "lost": lost,
                    "found": found,
                    "score": score,
                    "ket_luan": ket_luan,
                    "reason": reasons,
                    "details": details,
                    "score_image": anh_score,
                    "score_name": ten_score,
                    "score_color": mau_score,
                    "score_location": vi_tri_score,
                    "score_date": ngay_score,
                    "hinh_anh": anh_score,
                    "ten_do": ten_score,
                    "mau_sac": mau_score,
                    "vi_tri": vi_tri_score,
                    "ngay": ngay_score
                }

                heapq.heappush(matches_heap, (score, int(lost.id), int(found.id), result))
                if len(matches_heap) > result_limit:
                    heapq.heappop(matches_heap)

            except Exception as err:
                print(f"Bỏ qua cặp bài lỗi ({lost.id} - {found.id}): {err}")

    matches = [entry[3] for entry in sorted(matches_heap, reverse=True)]
    del matches_heap, lost_items, found_items
    gc.collect()
    return render_template("match.html", matches=matches)
