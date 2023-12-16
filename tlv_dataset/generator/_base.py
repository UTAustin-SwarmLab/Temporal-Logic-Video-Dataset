import abc


class DataGenerator(abc.ABC):
    """Data generator."""

    name: str
    save_dir: str

    @abc.abstractmethod
    def generate(self) -> any:
        """Generate data."""
