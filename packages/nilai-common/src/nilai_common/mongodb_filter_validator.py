from __future__ import annotations
from typing import Any, Dict, List, Union, Optional
from pydantic import BaseModel, Field, model_validator


class ComparisonOperator(BaseModel):
    """Model for MongoDB comparison operators"""

    eq: Optional[Any] = Field(None, alias="$eq")
    gt: Optional[Any] = Field(None, alias="$gt")
    gte: Optional[Any] = Field(None, alias="$gte")
    in_: Optional[List[Any]] = Field(None, alias="$in")
    lt: Optional[Any] = Field(None, alias="$lt")
    lte: Optional[Any] = Field(None, alias="$lte")
    ne: Optional[Any] = Field(None, alias="$ne")
    nin: Optional[List[Any]] = Field(None, alias="$nin")


class LogicalOperator(BaseModel):
    """Model for MongoDB logical operators"""

    and_: Optional[List[MongoFilter]] = Field(None, alias="$and")
    not_: Optional[Union[MongoFilter, Any]] = Field(None, alias="$not")
    nor: Optional[List[MongoFilter]] = Field(None, alias="$nor")
    or_: Optional[List[MongoFilter]] = Field(None, alias="$or")


class ElementOperator(BaseModel):
    """Model for MongoDB element operators"""

    exists: Optional[bool] = Field(None, alias="$exists")
    type: Optional[Union[int, str]] = Field(None, alias="$type")


class EvaluationOperator(BaseModel):
    """Model for MongoDB evaluation operators"""

    expr: Optional[Any] = Field(None, alias="$expr")
    jsonSchema: Optional[Dict[str, Any]] = Field(None, alias="$jsonSchema")
    mod: Optional[List[int]] = Field(None, alias="$mod")
    regex: Optional[str] = Field(None, alias="$regex")
    options: Optional[str] = Field(None, alias="$options")
    text: Optional[Union[str, Dict[str, Any]]] = Field(None, alias="$text")
    where: Optional[str] = Field(None, alias="$where")


class ArrayOperator(BaseModel):
    """Model for MongoDB array operators"""

    all: Optional[List[Any]] = Field(None, alias="$all")
    elemMatch: Optional[MongoFilter] = Field(None, alias="$elemMatch")
    size: Optional[int] = Field(None, alias="$size")


class BitwiseOperator(BaseModel):
    """Model for MongoDB bitwise operators"""

    bitsAllClear: Optional[Union[int, List[int]]] = Field(None, alias="$bitsAllClear")
    bitsAllSet: Optional[Union[int, List[int]]] = Field(None, alias="$bitsAllSet")
    bitsAnyClear: Optional[Union[int, List[int]]] = Field(None, alias="$bitsAnyClear")
    bitsAnySet: Optional[Union[int, List[int]]] = Field(None, alias="$bitsAnySet")


class GeoOperator(BaseModel):
    """Model for MongoDB geospatial operators"""

    geoIntersects: Optional[Dict[str, Any]] = Field(None, alias="$geoIntersects")
    geoWithin: Optional[Dict[str, Any]] = Field(None, alias="$geoWithin")
    near: Optional[Dict[str, Any]] = Field(None, alias="$near")
    nearSphere: Optional[Dict[str, Any]] = Field(None, alias="$nearSphere")


class MongoOperator(
    ComparisonOperator,
    LogicalOperator,
    ElementOperator,
    EvaluationOperator,
    ArrayOperator,
    BitwiseOperator,
    GeoOperator,
):
    """Combined model for all MongoDB operators"""

    pass


# MongoDB allows any field to have a value that's either:
# 1. A direct value (str, int, bool, etc.)
# 2. A dict with operators ($eq, $gt, etc.)
# 3. A dict with nested fields
# This makes validation complex since it's recursive and polymorphic


class MongoFilter(BaseModel):
    """
    MongoDB filter validator that handles all possible MongoDB query operators and structures.
    """

    model_config = {
        "extra": "allow",
        "arbitrary_types_allowed": True,
    }

    @model_validator(mode="before")
    @classmethod
    def validate_filter(cls, data):
        """
        Validate that the filter structure is correct.
        This is a recursive process since MongoDB filters can be nested.
        """
        if not isinstance(data, dict):
            return data

        for key, value in data.items():
            # Skip validation for logical operators like $and, $or which are handled by their own models
            if key.startswith("$"):
                continue

            # If value is a dict but doesn't contain any operator ($gt, $lt, etc.)
            # then it's a nested filter
            if isinstance(value, dict):
                # Check if it's an operator-based condition (e.g., {"$gt": 5})
                has_operators = any(k.startswith("$") for k in value.keys())

                if has_operators:
                    # Validate through MongoOperator
                    MongoOperator.model_validate(value)
                else:
                    # It's a nested filter, validate recursively
                    MongoFilter.model_validate(value)

            # If value is a list, validate each item in the list
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        MongoFilter.model_validate(item)

        return data


if __name__ == "__main__":
    example_filter = {
        "_id": {
            "$in": [
                "6e214b90-ea31-4d29-ae57-68af5f2cdc86",
                "688a3184-c2d7-4aed-ab2f-4d376a9b6a0d",
            ]
        }
    }

    valid = MongoFilter.model_validate(example_filter)
    print(f"Filter validation: {'Valid' if valid else 'Invalid'}")

    # Try more complex examples
    complex_filter = {
        "name": {"$regex": "^J", "$options": "i"},
        "age": {"$gte": 18, "$lt": 65},
        "$or": [{"status": "active"}, {"lastLogin": {"$gt": "2023-01-01"}}],
        "tags": {"$all": ["premium", "verified"]},
        "addresses.city": "New York",
    }

    valid = MongoFilter.model_validate(complex_filter)
    print(f"Complex filter validation: {'Valid' if valid else 'Invalid'}")
