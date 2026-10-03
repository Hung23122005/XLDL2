"""Run all four environments, reporting each failure independently."""
import json
from agent.agent import Agent


def main():
    results = {}
    for backend in ("onnx", "pytorch"):
        for device in ("cpu", "gpu"):
            name = f"agent_{backend}_{device}"
            agent = Agent(config=f"configs/{name}.yaml")
            print(f"=== {name} ===")
            try:
                agent.start()
                print(json.dumps(agent.get_info(), indent=2))
                agent.load_model("manifests/resnet18.yaml")
                x = agent.generate_input()
                for index in range(1, 4):
                    outputs = agent.predict(x)
                    assert outputs[0].shape == (1, 1000)
                    print(f"Predict {index}: output shapes = {[output.shape for output in outputs]}")
                report = agent.profile()
                assert report.is_file()
                results[name] = {"status": "PASS", "report": str(report)}
            except Exception as error:
                results[name] = {"status": "FAILED", "error": str(error)}
                print(f"{name} failed: {error}")
            finally:
                agent.stop()
    print(json.dumps(results, indent=2))
    return int(any(result["status"] == "FAILED" for result in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
