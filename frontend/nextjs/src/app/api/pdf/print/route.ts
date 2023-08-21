import { NextApiResponse } from "next";
import { NextResponse } from "next/server";
const puppeteer = require("puppeteer");

const printPDF = async () => {
  // Create a browser instance
  const browser = await puppeteer.launch({
    // headless: false,
    // args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  // Create a new page
  const page = await browser.newPage();

  // Website URL to export as pdf
  const website_url = "https://www.npmjs.com/package/puppeteer";
  //   "https://www.bannerbear.com/blog/how-to-download-images-from-a-website-using-puppeteer/";

  // Open URL in current page
  await page.goto(website_url, { waitUntil: "networkidle0" });

  //To reflect CSS used for screens instead of print
  await page.emulateMediaType("screen");

  // Downlaod the PDF
  const pdf = await page.pdf({
    //   path: "result.pdf",
    margin: { top: "100px", right: "50px", bottom: "100px", left: "50px" },
    printBackground: true,
    format: "A4",
  });

  // Close the browser instance
  await browser.close();
  return pdf;
};

// This handler will accept the url and fileName params
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const fileName = "my-resume.pdf";

  try {

    // Reference: 
    // https://reacthustle.com/blog/how-to-implement-download-files-in-nextjs-using-an-api-route#google_vignette
    // https://blog.logrocket.com/using-next-js-route-handlers/
    const pdf = await printPDF();
    return new Response(pdf, {
        headers: {
            "content-disposition": `attachment; filename="${fileName}"`,
            "Content-Type": "application/pdf"
        }
    });
  } catch (e: any) {
    console.error("Error is", e, __dirname);
    return new Response(JSON.stringify({
        error: e + ""
    }), {
        status: 500,
        headers: {
            "Content-Type": "application/json"
        },
    });
  }

  // //   return NextResponse.json(data);
}
