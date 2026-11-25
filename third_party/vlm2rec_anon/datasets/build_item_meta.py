import csv
import json

SEPERATOR = [',', ';']

def build_user_item_json(meta_csv, sequence_txt, output_json):
    item_meta = {}
    with open(meta_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["sid"]
            item_meta[sid] = {
                "title": row["title"],
                "imgurl": row["imgurl"]
            }

    user_data = {}
    with open(sequence_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user_id, item_sids = parts[0], parts[1:]
            
            titles = []
            imgurls = []
            for sid in item_sids:
                if sid in item_meta:
                    title = item_meta[sid]["title"]
                    for sep in SEPERATOR:
                        title = title.replace(sep, ' ')
                    titles.append(title)
                    imgurls.append(item_meta[sid]["imgurl"])
            user_data[user_id] = {
                "title": titles,
                "imgurl": imgurls
            }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(user_data, f, indent=2, ensure_ascii=False)



def filtering_csv_sep(csv_path, sep=SEPERATOR):
    filtered_rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row["title"]
            for s in sep:
                title = title.replace(s, ' ')
            row["title"] = title
            filtered_rows.append(row)
    
    with open(csv_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)
    


if __name__ == "__main__":
    import sys
    DATA_NAME = sys.argv[1]
    META_CSV = f"./processed/{DATA_NAME}_item_meta.csv"
    SEQ_TXT = f"./processed/{DATA_NAME}.txt"
    OUT_JSON = f"./processed/{DATA_NAME}_meta.json"

    filtering_csv_sep(META_CSV, sep=SEPERATOR)
    
    build_user_item_json(META_CSV, SEQ_TXT, OUT_JSON)
