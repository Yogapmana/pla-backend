import httpx
import asyncio

async def test():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Register a dummy user to get token
        res = await client.post("http://localhost:8000/api/v1/auth/register", json={
            "email": "testagent3@example.com",
            "password": "password123",
            "full_name": "Test Agent"
        })
        if res.status_code == 400: # already exists
            res = await client.post("http://localhost:8000/api/v1/auth/login", data={
                "username": "testagent3@example.com",
                "password": "password123"
            })
            if res.status_code != 200:
                print("Login failed:", res.text)
                return
        token = res.json()["access_token"]
            
        # Create a session
        res = await client.post("http://localhost:8000/api/v1/learning/sessions", headers={"Authorization": f"Bearer {token}"}, json={
            "topic": "Python Basics",
            "level": "beginner",
            "duration_weeks": 1,
            "hours_per_day": 1.0,
            "language": "id"
        })
        session_id = res.json()["id"]
        
        # Test chat
        res = await client.post("http://localhost:8000/api/v1/chat/message", headers={"Authorization": f"Bearer {token}"}, json={
            "session_id": session_id,
            "message": "hello"
        })
        print("Status:", res.status_code)
        print("Body:", res.text)

if __name__ == "__main__":
    asyncio.run(test())
