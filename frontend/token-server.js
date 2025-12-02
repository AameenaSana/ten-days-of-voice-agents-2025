const express = require("express");
const cors = require("cors");
const { AccessToken } = require("livekit-server-sdk");
const dotenv = require("dotenv");

dotenv.config({ path: "../backend/.env.local" });

const app = express();
app.use(cors());

app.get("/token", async (req, res) => {
  const { name } = req.query;
  if (!name) return res.status(400).json({ error: "Missing name" });

  const roomName = "improv_battle_room";

  const at = new AccessToken(
    process.env.LIVEKIT_API_KEY,
    process.env.LIVEKIT_API_SECRET,
    { identity: name }
  );

  at.addGrant({ roomJoin: true, room: roomName });
  const token = await at.toJwt();

  res.json({
    url: process.env.LIVEKIT_URL,
    token,
  });
});

const PORT = 8787;
app.listen(PORT, () => {
  console.log(`🎟️ Token server running on http://localhost:${PORT}`);
});