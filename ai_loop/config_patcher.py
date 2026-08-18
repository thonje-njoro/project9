"""
ai_loop/config_patcher.py — AST-safe config.py parameter updater.
Uses Python's ast module to parse and modify config values without corruption.
"""

import ast
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigPatcher:
    """
    AST-safe config.py parameter updater.
    Parses the file, finds the target value in the AST, replaces it,
    and writes back. This prevents syntax corruption.
    """

    def __init__(self, config_path: str | Path | None = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "backtest" / "config.py"
        self.config_path = Path(config_path)

    def read_config(self) -> str:
        """Read the config file."""
        return self.config_path.read_text()

    def write_config(self, content: str):
        """Write the config file."""
        self.config_path.write_text(content)

    def patch(self, strategy: str, symbol: str, parameter: str, value) -> bool:
        """
        Patch a single parameter in STRATEGY_PARAMS.

        Args:
            strategy: Strategy name (e.g., 'kalman_trend').
            symbol: Instrument symbol (e.g., 'GLD').
            parameter: Parameter name (e.g., 'Q').
            value: New value.

        Returns:
            True if successful.
        """
        try:
            source = self.read_config()
            tree = ast.parse(source)

            # Find and modify the target value
            patcher = _StrategyParamPatcher(strategy, symbol, parameter, value)
            new_tree = patcher.visit(tree)

            # Unparse back to source
            try:
                import astor
                new_source = astor.to_source(new_tree)
            except ImportError:
                # Fallback: use ast.unparse (Python 3.9+)
                new_source = ast.unparse(new_tree)

            # Write back
            self.write_config(new_source)
            logger.info(f"Patched {strategy}.{symbol}.{parameter} = {value}")
            return True

        except Exception as e:
            logger.error(f"Config patch failed: {e}")
            return False

    def get_current_value(self, strategy: str, symbol: str,
                          parameter: str) -> str | None:
        """
        Get current value of a parameter from config.
        Uses AST parsing for safety.
        """
        try:
            source = self.read_config()
            tree = ast.parse(source)

            finder = _StrategyParamFinder(strategy, symbol, parameter)
            finder.visit(tree)
            return finder.value

        except Exception as e:
            logger.error(f"Failed to read config value: {e}")
            return None


class _StrategyParamPatcher(ast.NodeTransformer):
    """AST transformer that patches STRATEGY_PARAMS values."""

    def __init__(self, strategy: str, symbol: str, parameter: str, value):
        self.strategy = strategy
        self.symbol = symbol
        self.parameter = parameter
        self.value = value
        self._in_strategy_params = False
        self._in_strategy = False
        self._in_symbol = False

    def visit_Assign(self, node):
        """Visit assignment nodes looking for STRATEGY_PARAMS."""
        # Check if this is STRATEGY_PARAMS = {...}
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "STRATEGY_PARAMS":
                self._in_strategy_params = True
                node.value = self._patch_dict(node.value, level="root")
                self._in_strategy_params = False
        return node

    def _patch_dict(self, node, level: str):
        """Recursively patch nested dict values."""
        if not isinstance(node, ast.Dict):
            return node

        for i, (key, value) in enumerate(zip(node.keys, node.values)):
            key_str = self._get_string_value(key)

            if level == "root" and key_str == self.strategy:
                # Found the strategy dict
                node.values[i] = self._patch_dict(value, level="strategy")

            elif level == "strategy" and key_str == self.symbol:
                # Found the symbol dict
                node.values[i] = self._patch_dict(value, level="symbol")

            elif level == "symbol" and key_str == self.parameter:
                # Found the parameter — replace value
                node.values[i] = self._make_value(self.value)
                logger.info(f"Found and replaced: {self.strategy}[{self.symbol}][{self.parameter}]")

        return node

    def _get_string_value(self, node) -> str | None:
        """Extract string value from an AST node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def _make_value(self, value):
        """Create an AST node for a Python value."""
        if isinstance(value, bool):
            return ast.Constant(value=value)
        elif isinstance(value, (int, float)):
            return ast.Constant(value=value)
        elif isinstance(value, str):
            return ast.Constant(value=value)
        else:
            return ast.Constant(value=str(value))


class _StrategyParamFinder(ast.NodeVisitor):
    """AST visitor that finds a specific parameter value."""

    def __init__(self, strategy: str, symbol: str, parameter: str):
        self.strategy = strategy
        self.symbol = symbol
        self.parameter = parameter
        self.value = None

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "STRATEGY_PARAMS":
                self._search_dict(node.value, level="root")

    def _search_dict(self, node, level: str):
        if not isinstance(node, ast.Dict):
            return

        for key, value in zip(node.keys, node.values):
            key_str = None
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                key_str = key.value

            if level == "root" and key_str == self.strategy:
                self._search_dict(value, level="strategy")
            elif level == "strategy" and key_str == self.symbol:
                self._search_dict(value, level="symbol")
            elif level == "symbol" and key_str == self.parameter:
                if isinstance(value, ast.Constant):
                    self.value = repr(value.value)
