"""Test script for the new text_generation endpoint."""

import requests
import json

BASE_URL = "http://localhost:19800/api/model_references/v1"


def test_endpoint(include_group: bool, endpoint: str) -> None:
    """Test the endpoint with the given parameters."""
    url = f"{BASE_URL}/{endpoint}"
    if endpoint == "text_generation":
        url += f"?include_group={str(include_group).lower()}"

    print(f"\n{'=' * 80}")
    print(f"Testing: {url}")
    print("=" * 80)

    response = requests.get(url)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        if data:
            # Get first model
            first_model_name = list(data.keys())[0]
            first_model = data[first_model_name]

            print(f"\nFirst Model: {first_model_name}")
            print(f"Has text_model_group: {'text_model_group' in first_model}")
            if "text_model_group" in first_model:
                print(f"text_model_group value: {first_model['text_model_group']}")

            print(f"\nModel fields: {list(first_model.keys())}")
            print(f"Total models: {len(data)}")
        else:
            print("No models returned (empty dict)")
    else:
        print(f"Error: {response.text}")


if __name__ == "__main__":
    print("Testing text generation endpoint with include_group parameter")

    # Test with include_group=false (default)
    test_endpoint(include_group=False, endpoint="text_generation")

    # Test with include_group=true
    test_endpoint(include_group=True, endpoint="text_generation")

    # Test the general endpoint for comparison
    print("\n" + "=" * 80)
    print("For comparison, testing general endpoint (should always have text_model_group):")
    print("=" * 80)
    test_endpoint(include_group=False, endpoint="text_generation")  # This uses the old route
