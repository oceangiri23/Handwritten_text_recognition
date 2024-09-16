'use client'
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { useState } from 'react';

export default function DropzoneUploader() {
    const [result , setResult] = useState('')    
    const [imagePreview, setImagePreview] = useState(null);
    const [uploadStatus, setUploadStatus] = useState('');
   
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: async (acceptedFiles) => {
      const file = acceptedFiles[0];
      const formData = new FormData();
      formData.append('file', file);
      const previewUrl:any = URL.createObjectURL(file);
      setImagePreview(previewUrl);
      setUploadStatus('Uploading...');

      try {
        const response = await axios.post(`http://localhost:8000/upload-image`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        const {prediction} = response.data
        setResult(prediction)
        setUploadStatus('Upload success');
        console.log('Upload success:', response.data);
      } catch (error) {
        setUploadStatus('Upload failed');
        console.error('Upload error:', error);
      }
    },
  });

  return (<>
  <div className='flex m-auto justify-center mt-9 flex-col ' style={{
    alignItems: 'center',
  }}>


    <div
      {...getRootProps()}
      style={{
        border: '2px dashed #cccccc',
        borderRadius: '4px',
        padding: '20px',
        textAlign: 'center',
        cursor: 'pointer',
        backgroundColor: isDragActive ? '#e0e0e0' : '#ffffff',
      }}
    >
      <input {...getInputProps()} />
      {isDragActive ? (
        <p>Drop the files here...</p>
      ) : (
        <p>Drag 'n' drop some files here, or click to select files</p>
      )}
    </div>
    {imagePreview && (
        <div style={{ marginTop: '20px', textAlign: 'center' }}>
          <img
            src={imagePreview}
            alt="Preview"
            style={{
              maxWidth: '100%',
              maxHeight: '400px',
              borderRadius: '4px',
            }}
          />
        </div>
      )}
      {uploadStatus && <p>{uploadStatus}</p>}
    {result && <div>{result}</div>}
  </div>

  </>

  );
}
