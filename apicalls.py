import json
from pathlib import Path

import requests


def run():
    # Specify a URL that resolves to your workspace
    URL = "http://127.0.0.1:8000"

    # Call each API endpoint and store the responses
    response1 = requests.get(f"{URL}/scoring").json()
    response2 = requests.get(f"{URL}/summarystats").json()
    response3 = requests.get(f"{URL}/diagnostics").json()
    response4 = requests.post(url=f"{URL}/prediction?filename=testdata/testdata.csv").json()

    # combine all API responses
    responses = {
        "score": response1["score"],
        "summary_statistics": response2["summary"],
        "missing_data": response3["missing_data"],
        "timing": response3["timing"],
        "packages": response3["packages"],
        "predictions": response4["predictions"]
    }

    # write the responses to your workspace
    with open('config.json', 'r') as f:
        config = json.load(f)
    api_returns_path = Path(config['output_model_path']) / Path("apireturns.txt")

    with open(api_returns_path, "w") as f:
        f.write(json.dumps(responses))


if __name__ == "__main__":
    run()
