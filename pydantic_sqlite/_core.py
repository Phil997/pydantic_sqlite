import importlib
import inspect
import json
import logging
import os
import sqlite3
import tempfile
import typing
from shutil import copyfile
from typing import Any, Dict, Generator, List, Literal, Union, get_origin

from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass
from pydantic.fields import FieldInfo
from sqlite_utils import Database as _SqliteDatabase

from ._misc import convert_value_into_union_types

SPECIALTYPE = [
    Any,
    Literal,
    Union]


class TableMetaInfo:
    """
    This class is a bucket which containes the meta informations about a table in a sqlite Database.

    Attributes:
        tablename `str`: The name of the table.
        model_class `ModelMetaclass`: The BaseModel class which will be used to initalize with a table entry.
        modulename: `str`: The name of the module which containes the model_class.
        classname: `str`: The correct name of the class.
        pks: `list[str]`: the primary key in the table.

    """
    def __init__(self, tablename: str, model_class: ModelMetaclass, pks: List[str]) -> None:
        self.tablename = tablename
        self.model_class = model_class
        self.modulename = self.model_class.__module__         # Get the module name
        self.classname = self.model_class.__name__              # Get the class name
        self.pks = pks  # TODO warum eine Liste??


class DataBase:
    known_models: Dict[str, TableMetaInfo]

    def __init__(self, **kwargs):
        self.known_models = dict()
        self._db = _SqliteDatabase(memory=True)

    def __call__(self, tablename) -> Generator[BaseModel, None, None]:
        """returns a Generator for all values in the Table. The returned values are subclasses of pydantic.BaseModel"""
        try:
            basemodel = self.known_models[tablename]
            foreign_refs = {key.column: key.other_table for key in self._db[tablename].foreign_keys}
        except KeyError:
            raise KeyError(f"can not find Table: {tablename} in Database") from None
        for row in self._db[tablename].rows:
            yield self._build_basemodel_from_dict(basemodel, row, foreign_refs)

    def add(
            self,
            tablename: str,
            value: BaseModel,
            foreign_tables={},
            update_nested_models=True,
            pk: str = "uuid") -> None:
        """adds a new value to the table tablename"""

        # unkown Tablename -> means new Table -> update the table basemodel ref list known_models
        if tablename not in self.known_models:
            self._add_model_to_known_models(tablename=tablename, model_class=value.__class__, pks=[pk])

        # check whether the value matches the basemodels in the table
        if not isinstance(value, BaseModel):
            msg = f"Can not add type '{type(value)}' to the table '{tablename}',"
            msg += f" which contains values of type '{self.known_models[tablename].basemodel_cls}'"
            raise ValueError(msg)

        # create dict for writing to the Table
        data_for_save = value.model_dump() if not hasattr(value, "sqlite_repr") else value.sqlite_repr

        foreign_keys = []
        for field_name, field in value.model_fields.items():
            field_value = getattr(value, field_name)

            if res := self._special_conversion(field_value):  # Special Insert with SQConfig.convert
                data_for_save[field_name] = res

            elif field.annotation == Any or get_origin(field.annotation) is Union:
                data_for_save[field_name] = field_value

            elif get_origin(field.annotation) is Literal:
                data_for_save[field_name] = str(field_value)

            elif get_origin(field.annotation) is list:
                obj = typing.get_args(field.annotation)[0]
                if inspect.isclass(obj) and issubclass(obj, BaseModel):
                    data_for_save[field_name] = [x.uuid for x in field_value]
                    foreign_table_name = self.get_check_foreign_table_name(field_name, foreign_tables)
                    foreign_keys.append((field_name, foreign_table_name, pk))
                else:
                    data_for_save[field_name] = [str(x) for x in field_value]

            elif issubclass(field.annotation, BaseModel):
                # the value has got a field which is of type BaseModel, so this filed must be in a foreign table
                # if the field is already in the Table it continues, but if is it not in the table it will add this
                # to the table recursive call to self.add
                foreign_table_name = self.get_check_foreign_table_name(field_name, foreign_tables)
                nested_obj_ids = self._upsert_value_in_foreign_table(
                    field_value,
                    foreign_table_name,
                    update_nested_models)
                data_for_save[field_name] = nested_obj_ids
                foreign_keys.append((field_name, foreign_table_name, pk))  # ignore=True

        self._db[tablename].upsert(data_for_save, pk=pk, foreign_keys=foreign_keys)

    def get_check_foreign_table_name(self, field_name: str,  foreign_tables: dict):
        if field_name not in foreign_tables.keys():
            keys = list(foreign_tables.keys())
            msg = f"detect field of Type BaseModel, but can not find '{field_name}'"
            msg += f"in foreign_tables (Keys: {keys})"
            raise KeyError(msg) from None
        else:
            foreign_table_name = foreign_tables[field_name]

        if foreign_table_name not in self._db.table_names():
            msg = f"Can not add a value, which has a foreign Key '{foreign_tables}'"
            msg += f" to a Table '{foreign_table_name}' which does not exists"
            raise KeyError(msg)
        return foreign_table_name

    def uuid_in_table(self, tablename: str, uuid: str) -> bool:
        """checks if the given uuid is used as a primary key in the table"""
        hits = [row for row in self._db[tablename].rows_where("uuid = ?", [uuid])]
        if len(hits) > 1:
            raise Exception("uuid is two times in table")  # TODO choice correct exceptiontype
        return False if not hits else True

    def value_in_table(self, tablename: str, value: BaseModel) -> bool:
        """checks if the given value is in the table"""
        return self.uuid_in_table(tablename, value.uuid)

    def value_from_table(self, tablename: str, uuid: str) -> typing.Any:
        """
        searchs the Objekt with the given uuid in the table and returns it.
        Returns a subclass of type pydantic.BaseModel
        """
        hits = [row for row in self._db[tablename].rows_where("uuid = ?", [uuid])]
        if len(hits) > 1:
            raise Exception("uuid is two times in table")  # TODO choice correct exceptiontype

        model = self.known_models[tablename]
        foreign_refs = {key.column: key.other_table for key in self._db[tablename].foreign_keys}
        return None if not hits else self._build_basemodel_from_dict(model, hits[0], foreign_refs=foreign_refs)

    def values_in_table(self, tablename) -> int:
        """returns the number of values in the Table"""
        return self._db[tablename].count

    def load(self, filename: str) -> None:
        """loads all data from the given file and adds them to the in-memory database"""
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Can not load {filename}")
        file_db = sqlite3.connect(filename)
        query = "".join(line for line in file_db.iterdump())
        self._db.conn.executescript(query)
        file_db.close()

        for row in self._db["__basemodels__"].rows:
            self._add_model_to_known_models(
                tablename=row['tablename'],
                model_class=getattr(importlib.import_module(row['modulename']), row['classname']),
                pks=json.loads(row['pks']))

    def save(self, filename: str) -> None:
        """saves alle values from the in_memory database to a file"""
        if not filename.endswith(".db"):
            filename += ".db"

        tmp_dir = tempfile.mkdtemp()
        name = filename.split(os.path.sep)[-1]
        tmp_name = tmp_dir + os.path.sep + name
        backup = tmp_dir + os.path.sep + "_backup.db"

        if os.path.isfile(filename):
            copyfile(filename, backup)
        try:
            file_db = sqlite3.connect(tmp_name)
            query = "".join(line for line in self._db.conn.iterdump())
            file_db.executescript(query)
            file_db.close()
            copyfile(tmp_name, filename)
        except Exception:
            logging.warning(f"saved the backup file under '{backup}'")
            raise

    def _add_model_to_known_models(self, tablename: str, model_class: ModelMetaclass, pks: List[str]):
        new_model = TableMetaInfo(tablename, model_class, pks)

        self.known_models[new_model.tablename] = new_model
        self._db["__basemodels__"].upsert(
            dict(
                tablename=new_model.tablename,
                modulename=new_model.modulename,
                classname=new_model.classname,
                pks=new_model.pks),
            pk="tablename")

    def _build_basemodel_from_dict(self, tablemeta: TableMetaInfo, row: dict, foreign_refs: dict):
        # returns a subclass object of type BaseModel which is build out of
        # class basemodel.basemodel_cls and the data out of the dict
        field_models: dict[str, FieldInfo] = tablemeta.model_class.model_fields
        d = {}

        for field_name, field_value in row.items():
            info = field_models[field_name]

            if field_name in foreign_refs.keys():  # the column contains another subclass of BaseModel
                if get_origin(info.annotation) == list:
                    data = [self.value_from_table(foreign_refs[field_name], val) for val in json.loads(field_value)]
                else:
                    data = self.value_from_table(foreign_refs[field_name], field_value)
            else:
                if get_origin(info.annotation) == list:
                    data = json.loads(field_value)
                elif get_origin(info.annotation) == Union:
                    data = convert_value_into_union_types(info.annotation, field_value)
                else:
                    data = field_value

            d.update({field_name: data})
        return tablemeta.model_class(**d)

    def _upsert_value_in_foreign_table(
            self,
            field_value,
            foreign_table_name,
            update_nested_models) -> Union[str, List[str]]:
        # The nested BaseModel will be inserted or upserted to the foreign table if it is not contained there,
        # or the update_nested_models parameter is True. If the value is Iterable (e.g. List) all values in the
        # List will be be inserted or upserted. The function returns the ids of the values

        # The foreign keys of this table are needed to add the nested basemodel object.
        foreign_refs = {key.column: key.other_table for key in self._db.table(foreign_table_name).foreign_keys}

        def add_nested_model(value):
            if not self.value_in_table(foreign_table_name, value) or update_nested_models:
                self.add(foreign_table_name, value, foreign_tables=foreign_refs)
            return value.uuid

        if not isinstance(field_value, List):
            return add_nested_model(field_value)
        else:
            return [add_nested_model(element) for element in field_value]

    def _special_conversion(self, field_value: Any) -> Union[bool, Any]:

        def special_possible(obj_class):
            try:
                if not hasattr(obj_class.SQConfig, 'convert'):
                    return False
                return True if obj_class.SQConfig.special_insert else False
            except AttributeError:
                return False

        if isinstance(field_value, List):
            if len(field_value) == 0:
                return False

            if not special_possible(obj_class := field_value[0].__class__):
                return False
            if not all(isinstance(value, type(field_value[0])) for value in field_value):
                raise ValueError(f"not all values in the List are from the same type: '{field_value}'")
            return [obj_class.SQConfig.convert(value) for value in field_value]
        else:
            if not special_possible(obj_class := field_value.__class__):
                return False
            return obj_class.SQConfig.convert(field_value)
