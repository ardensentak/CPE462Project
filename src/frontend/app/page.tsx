'use client';
import { useState } from 'react';
import { FaUpload } from 'react-icons/fa';

export default function Home() {
  const [image, setImage] = useState<File | null>(null); //will hold uploaded image files
  const [isRecyclable, setIsRecyclable] = useState<string | null>(null); //holds prediction result
  const [imageUrl, setImageUrl] = useState<string | null>(null); //will display image preview
  const [loading, setLoading] = useState(false); //to signal that the image classifier is loading

  //allows file input changes to occur + sends the files to backend for classification
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImage(file);
      setImageUrl(URL.createObjectURL(file));
      console.log('Uploaded file:', file);

      const formData = new FormData();
      formData.append('file', file);

      try {
        setLoading(true);
        const res = await fetch('/api/predict', {
          method: 'POST',
          body: formData,
        });
      
        if (!res.ok) {
          const errorText = await res.text();
          console.error('Backend responded with:', errorText);
          throw new Error('Prediction failed');
        }
      
        const data = await res.json();
        console.log(data); 
        setIsRecyclable(data.prediction); //updates result with the prediction
      } catch (err) {
        console.error('Prediction error:', err);
        setIsRecyclable('Error predicting'); //tells user if there was an error predicting
      } finally {
        setLoading(false);
      }
      
    }
  };

  //text on right side of website
  const part1 = 'Recyclable or not?';
  const part2 = 'Upload an image now to find out.';

  const createLetters = (text: string) => {
    return text.split('').map((letter, index) => {
      return letter === ' ' ? <span key={index}>&nbsp;</span> : letter;
    });
  };

  //website layout + style
  /* BASIC SUMMARY: 
  - 2 1/2 screen columns
  - left: allows image to be uploaded, shows image, outputs prediction
  - right: shows the text above
  */
  return (
    <main className="h-screen flex">
      <div className="w-1/2 flex flex-col justify-center items-center bg-gray-100 p-6 space-y-4">
        <label htmlFor="image-upload" className="text-3xl">
          Upload Image Here
        </label>
        <input
          id="image-upload"
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
        />
        <button
          onClick={() => document.getElementById('image-upload')?.click()}
          className="bg-blue-500 text-white py-4 px-6 rounded-lg flex items-center space-x-2"
        >
          <FaUpload />
          <span>Choose Image</span>
        </button>

        {imageUrl && (
          <img
            src={imageUrl}
            alt="Uploaded"
            className="mt-4 max-w-lg rounded shadow"
          />
        )}

        {loading && <p className="text-gray-600">Classifying image...</p>}
        {isRecyclable && !loading && (
          <p className="text-3xl font-bold text-green-700">Result: {isRecyclable}</p>
        )}
      </div>

      <div className="w-1/2 bg-green-100 p-6 flex items-center justify-center text-center">
        <div className="text-5xl leading-relaxed break-words whitespace-normal font-semibold">
          <div className="mb-1">
            {createLetters(part1).map((letter, index) => (
              <span
                key={index}
                className={`inline-block transition-opacity duration-300 delay-${index * 50} opacity-0 animate-appear text-shadow`}
                style={{ animationDelay: `${index * 50}ms` }}
              >
                {letter}
              </span>
            ))}
          </div>

          <div className="mt-2">
            {createLetters(part2).map((letter, index) => (
              <span
                key={index}
                className={`inline-block transition-opacity duration-300 delay-${(index + part1.length) * 50} opacity-0 animate-appear text-shadow`}
                style={{ animationDelay: `${(index + part1.length) * 50}ms` }}
              >
                {letter}
              </span>
            ))}
          </div>
        </div>
      </div>
    </main>
  );
}