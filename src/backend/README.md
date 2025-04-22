To run backend in the root of backend run the following:

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

This will run the backend on port 8000 so frontend must send requests to port 8000. 