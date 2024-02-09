import jwt
import time
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

in_file = open("certs/public.pem", "rb")
pem_bytes = in_file.read()
in_file.close()

public_key = serialization.load_pem_public_key(
    pem_bytes, backend=default_backend()
)

def validate_token(token):
    token = str.encode(token)
    data = jwt.decode(token, public_key, algorithms=["RS256"])
    if data != None:
        expires_at = float(data.get("expires_at"))
        now = time.time()
        if now > expires_at:
            return None
    return data
