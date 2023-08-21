import AWS from "aws-sdk";
import { NextRequest, NextResponse } from "next/server";
const credentials = new AWS.Credentials({
  accessKeyId: process.env.BHANZU_AWS_ACCESS_KEY_ID || "",
  secretAccessKey: process.env.BHANZU_AWS_SECRET_KEY || "",
});

const IvsChat = new AWS.Ivschat({ region: "us-east-1", credentials });

const createToken = async (
  userId: string,
  chatroomARN: string,
  isAdmin: boolean
) => {
  let token;

  const capabilities = ["SEND_MESSAGE"];
  if (isAdmin) {
    capabilities.push(...["DISCONNECT_USER", "DELETE_MESSAGE"]);
  }

  const params = {
    roomIdentifier: chatroomARN,
    userId,
    attributes: {},
    capabilities: capabilities,
    sessionDurationInMinutes: 60,
  };
  try {
    const data = await IvsChat.createChatToken(params).promise();
    console.log("create token data", data);
    token = data.token;
  } catch (e: any) {
    console.error(e);
    throw new Error(e);
  }
  return token;
};

export async function GET(request: Request, res: Response) {
  const { searchParams } = new URL(request.url);
  const userId = searchParams.get("userId");
  const chatroomARN = searchParams.get("chatroomARN");
  console.log(
    "get request",
    JSON.stringify(NextRequest),
    JSON.stringify({ ...request })
  );
  console.log("get request", searchParams, userId, chatroomARN);

  const data: any = {};
  if (userId && chatroomARN) {
    const token = await createToken(userId, chatroomARN);
    data.token = token;
  }

  return NextResponse.json({ data });

  // //   return NextResponse.json(data);
}

// create-token-api
// http://localhost:4000/api/aws/create-token?userId=girish1&chatroomARN=arn:aws:ivschat:us-east-1:985105535842:room/NBHDZCSbKoGV
