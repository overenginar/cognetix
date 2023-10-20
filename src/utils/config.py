import configparser


class Config:
    def __init__(self, file_name: str = "config.ini") -> None:
        self.file_name = file_name
        self.config = configparser.ConfigParser()
        self.config.read(self.file_name)
