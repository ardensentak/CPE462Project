import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Header from "../components/ui/header"; 
import Footer from "../components/ui/footer"; 

//added so I can use the Inter font
const inter = Inter({subsets: ["latin"]}); 

//browser tab title and description
export const metadata: Metadata = {
  title: "Image Classifier",
  description: "Recyclable or Not?",
};

//added Header and Footer into the layout so that they would show on my webpage 
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={inter.className}
      >
        <Header/> 
        {children}
        <Footer/>
      </body>
    </html>
  );
}
