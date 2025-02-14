from typing import Optional

def require(test: bool, throwMessage: Optional[str] = None):
    if not test:
        if throwMessage is not None:
            raise Exception(throwMessage)
        else:
            raise Exception("Requirement Failed.")


