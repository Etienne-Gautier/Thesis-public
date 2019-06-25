from DbAccess.DbConnect import DbConnect
import bson
import datetime
from typing import Dict, Any, List, Iterator
from synthia_example import SynthiaExample


class SynthiaDao(DbConnect):

    def __init__(self):
        super().__init__()
        self.synthia_collection = self.db["Synthia database"]

    @staticmethod
    def convert_synthia_to_bson(item: SynthiaExample) -> Dict[str, Any]:
        return {"path": item.path,
                "is_training": item.is_training,
                "is_raw_data": item.is_raw_data,
                "sparsity_level": bson.Decimal128(item.sparsity_level),
                "sequence": item.sequence,
                "is_right_camera": item.is_right_camera,
                "camera_index": item.camera_index,
                "order_in_sequence": item.order_in_sequence,
                "timestamp": int(item.timestamp),
                "created_at": datetime.datetime.utcnow()}

    @staticmethod
    def convert_bson_to_synthia(record: Dict[str, Any]) -> SynthiaExample:
        image_obj: SynthiaExample = SynthiaExample(record["path"])
        image_obj.id = record["_id"]
        image_obj.is_training = record["is_training"]
        image_obj.is_raw_data = record["is_raw_data"]
        image_obj.sparsity_level = record["sparsity_level"].to_decimal()
        image_obj.sequence = record["sequence"]
        image_obj.is_right_camera = record["is_right_camera"]
        image_obj.camera_index = record["camera_index"]
        image_obj.order_in_sequence = record["order_in_sequence"]
        image_obj.timestamp = record["timestamp"]
        return image_obj

    def insert_synthia_example(self, record: SynthiaExample):
        return self.synthia_collection.insert_one(SynthiaDao.convert_synthia_to_bson(record))


    def get_Synthia_train_sequences(self) -> List[str]:
        return self.synthia_collection.distinct("sequence", {"is_training": True})
    
    def get_Synthia_eval_sequences(self) -> List[str]:
        return self.synthia_collection.distinct("sequence", {"is_training": False})

    def get_Synthia_images_from_sequence(self, sequence: str, is_right_camera: bool = False, camera_index: int=1)-> List[SynthiaExample]:
        return [ SynthiaDao.convert_bson_to_synthia(record) for record in
        self.synthia_collection.find({"sequence": sequence, "is_right_camera": is_right_camera, "camera_index": camera_index}).sort("order_in_sequence")]

    def get_all_Synthia_images_from_sequence(self, sequence: str)-> List[SynthiaExample]:
        return [ SynthiaDao.convert_bson_to_synthia(record) for record in self.synthia_collection.find({"sequence": sequence})]
    
    def update_Synthia_example(self, item: SynthiaExample) -> None:
        self.synthia_collection.update_one(
            {"_id": item.id},
            {"$set": SynthiaDao.convert_synthia_to_bson(item)}
        )

