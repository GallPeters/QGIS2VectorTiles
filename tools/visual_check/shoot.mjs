// shoot.mjs <browser> <port> <out.png> <x3857> <y3857> <zoom> <width> <height>
// A MapLibre screenshot of an export served on localhost:<port>, at a
// centre in EPSG:3857 and a zoom, with the controls hidden, after the map
// has loaded its tiles. Prints the page's errors and warnings, if any.
import puppeteer from 'puppeteer-core';

const [browser, port, out, x, y, zoom, width, height] = process.argv.slice(2);
const lon = (x / 6378137) * (180 / Math.PI);
const lat = (2 * Math.atan(Math.exp(y / 6378137)) - Math.PI / 2) * (180 / Math.PI);
const instance = await puppeteer.launch({ executablePath: browser, headless: 'new' });
const page = await instance.newPage();
await page.setViewport({ width: +width, height: +height, deviceScaleFactor: 1 });
const messages = [];
page.on('pageerror', (e) => messages.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') messages.push(m.text().slice(0, 200)); });
await page.goto(`http://localhost:${port}/?viewer=ml&z=${zoom}&c=${lon.toFixed(7)},${lat.toFixed(7)}&basemap=none`,
    { waitUntil: 'networkidle0', timeout: 120000 });
await page.addStyleTag({ content: '.maplibregl-control-container, [class*=q2vt-top], [class*=q2vt-bottom], .q2vt-panel, .q2vt-coords { display: none !important; }' });
await new Promise((resolve) => setTimeout(resolve, 3000));
await page.screenshot({ path: out });
if (messages.length) console.log([...new Set(messages)].slice(0, 5).join(' | '));
await instance.close();
process.exit(0);
