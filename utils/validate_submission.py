import csv
import os
import sys

# Constants
EXPECTED_COLUMNS = 12
MAX_CAPTION_LENGTH = 9000
EXPECTED_HEADER = [
    'query_id', 'article_id_1', 'article_id_2', 'article_id_3', 'article_id_4',
    'article_id_5', 'article_id_6', 'article_id_7', 'article_id_8', 'article_id_9',
    'article_id_10', 'generated_caption'
]

def check_blank_lines(file_path):
    """Kiểm tra xem tệp có dòng trống hay không."""
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                print(f"LỖI: Dòng trống được tìm thấy ở dòng {i}.")
                return False
    print("✓ Không tìm thấy dòng trống.")
    return True

def validate_csv(file_path, official_query_ids_path=None):
    """
    Xác thực tệp CSV dựa trên các quy tắc đã cho.
    :param file_path: Đường dẫn đến tệp submission.csv.
    :param official_query_ids_path: (Tùy chọn) Đường dẫn đến tệp chứa các query_id chính thức.
    """
    print(f"Bắt đầu xác thực tệp: {file_path}")

    if not os.path.exists(file_path):
        print(f"LỖI: Không tìm thấy tệp '{file_path}'.")
        return

    # 1. Kiểm tra dòng trống
    if not check_blank_lines(file_path):
        return # Dừng lại nếu có dòng trống

    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            # Sử dụng sniffing để kiểm tra delimiter, nhưng ở đây ta mặc định là dấu phẩy
            # dialect = csv.Sniffer().sniff(csvfile.read(1024))
            # csvfile.seek(0)
            # reader = csv.reader(csvfile, dialect)
            reader = csv.reader(csvfile, delimiter=',')
            
            # 2. Kiểm tra header
            header = next(reader, None)
            if not header:
                print("LỖI: Tệp CSV trống hoặc không có header.")
                return
            
            # So sánh header với header chuẩn
            if header != EXPECTED_HEADER:
                print("LỖI: Header của tệp CSV không đúng định dạng.")
                print(f"  - Header mong muốn: {EXPECTED_HEADER}")
                print(f"  - Header trong tệp: {header}")
                return
            print("✓ Header của tệp CSV hợp lệ.")

            # 3. Đọc danh sách query_id chính thức (nếu có)
            official_query_ids = set()
            if official_query_ids_path:
                if os.path.exists(official_query_ids_path):
                    try:
                        with open(official_query_ids_path, 'r', newline='', encoding='utf-8') as f:
                            reader = csv.reader(f)
                            next(reader) # Bỏ qua header
                            official_query_ids = {row[0].strip() for row in reader if row and row[0].strip()}
                        print(f"✓ Đã tải {len(official_query_ids)} query_id chính thức từ '{official_query_ids_path}'.")
                    except StopIteration:
                        print(f"CẢNH BÁO: Tệp query_id chính thức '{official_query_ids_path}' trống hoặc chỉ có header.")
                        official_query_ids = set()
                    except Exception as e:
                        print(f"LỖI: Không thể đọc tệp query_id chính thức '{official_query_ids_path}'. Lỗi: {e}")
                        official_query_ids = set() # Đặt lại khi có lỗi
                else:
                    print(f"CẢNH BÁO: Không tìm thấy tệp query_id chính thức tại '{official_query_ids_path}'. Sẽ bỏ qua bước kiểm tra query_id.")

            # 4. Kiểm tra từng dòng dữ liệu
            seen_query_ids = set()
            errors = []
            for row_num, row in enumerate(reader, 2): # Bắt đầu từ dòng 2
                # Kiểm tra số lượng cột
                if len(row) != EXPECTED_COLUMNS:
                    errors.append(f"Dòng {row_num}: Số cột không hợp lệ. Mong muốn {EXPECTED_COLUMNS}, nhận được {len(row)}.")
                    continue # Chuyển sang dòng tiếp theo nếu cấu trúc đã sai

                query_id, *article_ids, caption = row
                
                # Kiểm tra query_id
                if not query_id.strip():
                     errors.append(f"Dòng {row_num}: query_id không được để trống.")
                elif query_id in seen_query_ids:
                    errors.append(f"Dòng {row_num}: query_id '{query_id}' bị lặp lại.")
                else:
                    seen_query_ids.add(query_id)
                
                if official_query_ids and query_id not in official_query_ids:
                    errors.append(f"Dòng {row_num}: query_id '{query_id}' không có trong danh sách chính thức.")

                # Kiểm tra article_ids (cột 2-11)
                for i, art_id in enumerate(article_ids, 1):
                    if not (art_id.strip().isdigit() or art_id.strip() == '#'):
                        errors.append(f"Dòng {row_num}, Cột {i+1}: article_id '{art_id}' không hợp lệ. Phải là một số hoặc '#'.")

                # Kiểm tra caption
                if not (caption.startswith('"') and caption.endswith('"')):
                    errors.append(f"Dòng {row_num}: Caption phải được đặt trong dấu ngoặc kép.")
                
                # Bỏ dấu ngoặc kép để kiểm tra độ dài
                caption_content = caption[1:-1]
                if len(caption_content) > MAX_CAPTION_LENGTH:
                    errors.append(f"Dòng {row_num}: Caption vượt quá độ dài tối đa {MAX_CAPTION_LENGTH} ký tự (hiện tại: {len(caption_content)}).")

            if errors:
                print("\nTìm thấy các lỗi sau trong tệp CSV:")
                for error in errors:
                    print(f"  - {error}")
            else:
                print("✓ Tất cả các dòng dữ liệu đều hợp lệ.")
                print("\n-------------------------------------------")
                print(">>> Xác thực thành công! Tệp 'submission.csv' có vẻ hợp lệ. <<<")
                print("\nLưu ý quan trọng:")
                print("1. Hãy đảm bảo rằng các `article_id` tồn tại trong cơ sở dữ liệu (script này chỉ kiểm tra định dạng).")
                print("2. Nén tệp `submission.csv` thành `submission.zip` trước khi nộp.")
                print("3. KHÔNG đặt tệp `submission.csv` vào trong một thư mục con trước khi nén.")

    except csv.Error as e:
        print(f"LỖI: Không thể đọc tệp CSV. Lỗi: {e}")
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Sử dụng: python validate_submission.py <đường_dẫn_tới_submission.csv> [đường_dẫn_tới_query_ids.csv]")
        sys.exit(1)
        
    csv_file_path = sys.argv[1]
    query_ids_file = sys.argv[2] if len(sys.argv) > 2 else None

    validate_csv(csv_file_path, query_ids_file) 