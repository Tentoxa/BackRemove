import os
import unittest

os.environ.setdefault("API_KEY", "test-only-key")

from app.main import app


class ApiContractTests(unittest.TestCase):
    def test_remove_bg_exposes_fast_and_quality_query_modes(self):
        operation = app.openapi()["paths"]["/remove-bg"]["post"]
        model_parameter = next(
            parameter
            for parameter in operation["parameters"]
            if parameter["name"] == "model"
        )
        schema = model_parameter["schema"]
        model_schema = app.openapi()["components"]["schemas"][
            schema["$ref"].split("/")[-1]
        ]

        self.assertEqual(model_parameter["in"], "query")
        self.assertEqual(schema["default"], "fast")
        self.assertEqual(model_schema["enum"], ["fast", "quality"])


if __name__ == "__main__":
    unittest.main()
