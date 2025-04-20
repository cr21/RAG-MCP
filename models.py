from pydantic import BaseModel
from typing import Optional, Dict, List
import re

class ProductMetadata(BaseModel):
    id: int
    price: float
    discountedPrice: float
    styleType: str
    productTypeId: int
    articleNumber: str
    productDisplayName: str
    variantName: str
    myntraRating: float
    catalogAddDate: int
    brandName: str
    ageGroup: str
    gender: str
    baseColour: str
    colour1: Optional[str]
    colour2: Optional[str]
    fashionType: str
    season: str
    year: str
    usage: str
    vat: float
    displayCategories: str

class ProductResponse(BaseModel):
    id: int
    product_display_name: str
    display_categories: str
    product_descriptors: dict
    article_attributes: dict
    master_category: dict
    sub_category: dict
    article_type: dict
    product_metadata: ProductMetadata

class ProductChunkTyped(BaseModel):
    """
    Pydantic model for product chunk
    """
    id: int
    product_content: str
    metadata: ProductMetadata

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing HTML tags and special characters
        """
        if not text:
            return ""
            
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Replace common HTML entities
        html_entities = {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '&ndash;': '-',
            '&mdash;': '-',
            '&bull;': '•',
        }
        for entity, replacement in html_entities.items():
            text = text.replace(entity, replacement)
        
        # Replace any remaining HTML entities (like &#123;)
        text = re.sub(r'&#\d+;', '', text)
        text = re.sub(r'&[a-zA-Z0-9]+;', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    @classmethod
    def from_json(cls, data: dict) -> "ProductChunkTyped":
        """
        Factory method to create ProductChunkTyped from json data
        """
        product_data = data.get("data", {})
        id = product_data.get("id")
        
        # Handle article attributes with cleaned text
        article_attributes = ", ".join([
            f"{k}: {cls.clean_text(str(v))}" 
            for k, v in product_data.get("articleAttributes", {}).items()
        ])

        # Handle master category dictionary with cleaned text
        master_category_dict = product_data.get("masterCategory", {})
        master_category_attrs = ", ".join([
            f"{k}: {cls.clean_text(str(v))}" 
            for k, v in master_category_dict.items()
            if not isinstance(v, dict) and v is not None
        ])

        # Handle sub category dictionary with cleaned text
        sub_category_dict = product_data.get("subCategory", {})
        sub_category_attrs = ", ".join([
            f"{k}: {cls.clean_text(str(v))}" 
            for k, v in sub_category_dict.items()
            if not isinstance(v, dict) and v is not None
        ])

        # Handle article type dictionary with cleaned text
        article_type_dict = product_data.get("articleType", {})
        article_type_attrs = ", ".join([
            f"{k}: {cls.clean_text(str(v))}" 
            for k, v in article_type_dict.items()
            if not isinstance(v, dict) and v is not None
        ])
        
        # Handle product descriptors with cleaned text
        product_descriptors = []
        for desc_key, desc_data in product_data.get("productDescriptors", {}).items():
            if isinstance(desc_data, dict) and "value" in desc_data:
                value = cls.clean_text(desc_data["value"])
                if value:  # Only add if there's actual content after cleaning
                    product_descriptors.append(f"{desc_key}: {value}")

        # Construct product_content string with cleaned text
        product_content = (
            f"productDisplayName: {cls.clean_text(product_data.get('productDisplayName'))}. "
            f"displayCategories: {cls.clean_text(product_data.get('displayCategories'))}. "
            f"Product Descriptors: {', '.join(product_descriptors)}. "
            f"Article Attributes: {article_attributes}. "
            f"Master Category: {master_category_attrs}. "
            f"Sub Category: {sub_category_attrs}. "
            f"Article Type: {article_type_attrs}. "
        )

        # Construct metadata
        metadata = {
            "id": product_data.get("id"),
            "price": product_data.get("price"),
            "discountedPrice": product_data.get("discountedPrice"),
            "styleType": product_data.get("styleType"),
            "productTypeId": product_data.get("productTypeId"),
            "articleNumber": product_data.get("articleNumber"),
            "productDisplayName": product_data.get("productDisplayName"),
            "variantName": product_data.get("variantName"),
            "myntraRating": product_data.get("myntraRating"),
            "catalogAddDate": product_data.get("catalogAddDate"),
            "brandName": product_data.get("brandName"),
            "ageGroup": product_data.get("ageGroup"),
            "gender": product_data.get("gender"),
            "baseColour": product_data.get("baseColour"),
            "colour1": product_data.get("colour1"),
            "colour2": product_data.get("colour2"),
            "fashionType": product_data.get("fashionType"),
            "season": product_data.get("season"),
            "year": product_data.get("year"),
            "usage": product_data.get("usage"),
            "vat": product_data.get("vat"),
            "displayCategories": product_data.get("displayCategories")
        }

        return cls(
            id=id,
            product_content=product_content,
            metadata=ProductMetadata(**metadata)
        )


class AddInput(BaseModel):
    a: int
    b: int

class AddOutput(BaseModel):
    result: int

class SqrtInput(BaseModel):
    a: int

class SqrtOutput(BaseModel):
    result: float

class StringsToIntsInput(BaseModel):
    string: str

class StringsToIntsOutput(BaseModel):
    ascii_values: List[int]

class ExpSumInput(BaseModel):
    int_list: List[int]

class ExpSumOutput(BaseModel):
    result: float


# if __name__ == "__main__":
#     import json
#     from pprint import pprint
#     with open("/Users/chiragtagadiya/MyProjects/EAG1/RAG-MCP/documents/1163.json", "r") as f:
#         data = json.load(f)
#     product = ProductChunkTyped.from_json(data)
#     json_formatted_str = json.dumps(product.model_dump(), indent=2)
#     print(json_formatted_str)
#     # pprint(product)