import requests
import time
import subprocess

def test_auth():
    # Start server
    server_process = subprocess.Popen(["./venv/bin/uvicorn", "app.main:app", "--port", "8001"])
    time.sleep(3) # wait for server to start

    try:
        base_url = "http://localhost:8001/api/v1/auth"
        
        # Test 1: Register
        print("Testing Registration...")
        reg_resp = requests.post(f"{base_url}/register", json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "password123"
        })
        print(f"Register status: {reg_resp.status_code}")
        if reg_resp.status_code == 201:
            print("Registration OK")
        elif reg_resp.status_code == 400:
            print("User already exists (this is fine for repeated runs)")
        else:
            print(f"Failed: {reg_resp.text}")

        # Test 2: Login
        print("\nTesting Login...")
        login_resp = requests.post(f"{base_url}/login", data={
            "username": "test@example.com",
            "password": "password123"
        })
        print(f"Login status: {login_resp.status_code}")
        token = ""
        if login_resp.status_code == 200:
            token = login_resp.json().get("access_token")
            print("Login OK")
        else:
            print(f"Failed: {login_resp.text}")

        # Test 3: Get Me
        if token:
            print("\nTesting Get Me...")
            me_resp = requests.get(f"{base_url}/me", headers={
                "Authorization": f"Bearer {token}"
            })
            print(f"Me status: {me_resp.status_code}")
            if me_resp.status_code == 200:
                print(f"User Data: {me_resp.json()}")
            else:
                print(f"Failed: {me_resp.text}")

    finally:
        server_process.terminate()

if __name__ == "__main__":
    test_auth()
