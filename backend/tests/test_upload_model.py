# проверка загрузки файла неправильного формата

from io import BytesIO

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_upload_wrong_file_format():
    files = {
        "file": (
            "model.txt",
            BytesIO(b"not a model"),
            "text/plain"
        )
    }

    response = client.post("/upload-model", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "Файл должен быть .pkl"