"""Environment-created ASGI entrypoint for the local supplier A2A peer."""

from industrial_ops_agent.supplier_sandbox.app import SupplierSandboxSettings, create_app

app = create_app(SupplierSandboxSettings.from_environment())
