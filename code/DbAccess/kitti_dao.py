from DbAccess.DbConnect import DbConnect
from datetime import datetime
from typing import Dict, Any, List
from KITTIExample import KITTIExample

class KITTIDao(DbConnect):

    def __init__(self):
        super().__init__()
        self.KITTI_collection = self.db["KITTI database"]


    @staticmethod
    def convert_KITTI_to_bson(record: KITTIExample) -> Dict[str, Any]:
        return {"path": record.path,
            "is_training": record.is_training,
            "is_raw_data": record.is_raw_data,
            "sequence": record.sequence,
            "is_right_camera": record.is_right_camera,
            "order_in_sequence": record.order_in_sequence,
            "timestamp": int(record.timestamp),
            "created_at": datetime.utcnow()}
    @staticmethod
    def convert_bson_to_KITTI(record: Dict[str, Any]) -> KITTIExample:
        image_obj : KITTIExample = KITTIExample(record["path"])
        image_obj.id = record["_id"]
        image_obj.is_training = record["is_training"]
        image_obj.is_raw_data = record["is_raw_data"]
        image_obj.sequence = record["sequence"]
        image_obj.is_right_camera = record["is_right_camera"]
        image_obj.order_in_sequence = record["order_in_sequence"]
        image_obj.timestamp = record["timestamp"]
        return image_obj

    def insert_KITTI_example(self, record: KITTIExample):
        return self.KITTI_collection.insert_one(KITTIDao.convert_KITTI_to_bson(record))

    def get_KITTI_train_sequences(self) -> List[str]:
        return self.KITTI_collection.distinct("sequence", {"is_training": True})
    
    def get_KITTI_eval_sequences(self) -> List[str]:
        return self.KITTI_collection.distinct("sequence", {"is_training": False})

    def get_KITTI_images_from_sequence(self, sequence: str, is_raw_data: bool,is_right_camera: bool = False)-> List[KITTIExample]:
        return [ KITTIDao.convert_bson_to_KITTI(record) for record in self.KITTI_collection.find({"sequence": sequence, "is_right_camera": is_right_camera, "is_raw_data": is_raw_data})] #add .sort("order_in_sequence") to result of finds

    def get_KITTI_all(self) -> List[KITTIExample]:
        return [KITTIDao.convert_bson_to_KITTI(record) for record in self.KITTI_collection.find()]
    
    def update_KITTI_example(self, item: KITTIExample):
        self.KITTI_collection.update_one(
            {"_id": item.id},
            {"$set": KITTIDao.convert_KITTI_to_bson(item)}
        )
    