"""
cloudinary_helper.py - Cloudinary integration for image storage

Hỗ trợ upload ảnh lên Cloudinary cloud storage.
Miễn phí: 25GB storage, 25GB bandwidth/tháng
"""

import cloudinary
import cloudinary.uploader
from pathlib import Path
import os


class CloudinaryHelper:
    """Helper class để tương tác với Cloudinary"""
    
    def __init__(self, cloud_name, api_key, api_secret):
        """
        Khởi tạo Cloudinary config
        
        Args:
            cloud_name: Cloudinary cloud name
            api_key: Cloudinary API key
            api_secret: Cloudinary API secret
        """
        self.cloud_name = cloud_name
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Configure Cloudinary
        cloudinary.config(
            cloud_name="dqupovatf",
            api_key="618168733961471",
            api_secret="f7EuWrh0XgVeokAAwj1MwCJS000"
        )
    
    def upload_image(self, file_path, folder='clothing_classifier'):
        """
        Upload ảnh lên Cloudinary
        
        Args:
            file_path: Đường dẫn file ảnh
            folder: Folder trong Cloudinary
            
        Returns:
            dict: {success, url, error}
        """
        try:
            result = cloudinary.uploader.upload(
                file_path,
                folder=folder,
                resource_type='auto',
                overwrite=False,
                unique_filename=True
            )
            
            return {
                'success': True,
                'url': result['secure_url'],
                'public_id': result['public_id']
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'url': None
            }
    
    def delete_image(self, public_id):
        """
        Xoá ảnh từ Cloudinary
        
        Args:
            public_id: Public ID của ảnh
            
        Returns:
            dict: {success, message}
        """
        try:
            result = cloudinary.uploader.destroy(public_id)
            return {
                'success': True,
                'message': 'Deleted successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'message': str(e)
            }


def create_cloudinary_helper(cloud_name, api_key, api_secret):
    """
    Factory function để tạo CloudinaryHelper instance
    
    Args:
        cloud_name: Cloudinary cloud name
        api_key: Cloudinary API key
        api_secret: Cloudinary API secret
        
    Returns:
        CloudinaryHelper: Instance sẵn sàng dùng
    """
    return CloudinaryHelper(cloud_name, api_key, api_secret)


# Ví dụ cách dùng:
if __name__ == '__main__':
    # Get credentials từ environment variables
    cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME', 'your_cloud_name')
    api_key = os.getenv('CLOUDINARY_API_KEY', 'your_api_key')
    api_secret = os.getenv('CLOUDINARY_API_SECRET', 'your_api_secret')
    
    # Tạo helper
    helper = create_cloudinary_helper(cloud_name, api_key, api_secret)
    
    # Upload ảnh
    # result = helper.upload_image('path/to/image.jpg')
    # print(result)
