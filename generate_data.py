import requests
import json
import os
import argparse

def fetch_data(offset, length=100):
    url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": "kkcosmos/instagram-images-with-captions",
        "config": "default",
        "split": "train",
        "offset": offset,
        "length": length
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to retrieve data at offset {offset}. Status code: {response.status_code}")
        return None

def main(output_dir):
    total_images = 200
    batch_size = 100
    all_rows = []  # Store all rows from each batch here

    for offset in range(0, total_images, batch_size):
        data = fetch_data(offset, batch_size)
        if data is not None and 'rows' in data:
            all_rows.extend(data['rows'])  # Extract and append the 'rows' from each batch
        else:
            break

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, 'bigdata.json')
    with open(output_file, 'w') as f:
        json.dump(all_rows, f)  # Save the combined rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch data and save to JSON file.")
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the output JSON file.')
    args = parser.parse_args()

    main(args.output_dir)