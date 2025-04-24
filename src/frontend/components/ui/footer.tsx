//creates custom website footer
//this footer is designed to mention the current year and my name on my website
import React from 'react';

const Footer = () => {
  return (
    <footer className="text-center text-sm text-gray-500 py-2 mt-2">
      <p>&copy; {new Date().getFullYear()} Arden Sentak</p>
    </footer>
  );
};

export default Footer;
