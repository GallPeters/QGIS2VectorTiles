// Geodesic measurements on WGS84 longitude/latitude coordinates, and their
// display. Lengths use the haversine formula; areas the spherical-excess
// method (as in Turf.js), both on a sphere of the mean Earth radius.

const EARTH_RADIUS = 6371008.8;
const rad = (deg) => (deg * Math.PI) / 180;

export function distance([lon1, lat1], [lon2, lat2]) {
    const dLat = rad(lat2 - lat1);
    const dLon = rad(lon2 - lon1);
    const a = Math.sin(dLat / 2) ** 2
        + Math.cos(rad(lat1)) * Math.cos(rad(lat2)) * Math.sin(dLon / 2) ** 2;
    return 2 * EARTH_RADIUS * Math.asin(Math.min(1, Math.sqrt(a)));
}

export function lineLength(coordinates) {
    let total = 0;
    for (let i = 1; i < coordinates.length; i++) {
        total += distance(coordinates[i - 1], coordinates[i]);
    }
    return total;
}

function ringArea(ring) {
    const n = ring.length - 1;  // Rings are closed: the last point repeats the first.
    if (n <= 2) return 0;
    let total = 0;
    for (let i = 0; i < n; i++) {
        const lower = ring[i];
        const middle = ring[i + 1 === n ? 0 : i + 1];
        const upper = ring[i + 2 >= n ? (i + 2) % n : i + 2];
        total += (rad(upper[0]) - rad(lower[0])) * Math.sin(rad(middle[1]));
    }
    return Math.abs((total * EARTH_RADIUS * EARTH_RADIUS) / 2);
}

export function polygonArea(rings) {
    if (!rings.length) return 0;
    return Math.max(0, rings.slice(1).reduce((area, hole) => area - ringArea(hole), ringArea(rings[0])));
}

export function formatLength(meters) {
    if (meters < 1000) return `${meters.toFixed(meters < 10 ? 2 : 1)} m`;
    return `${(meters / 1000).toFixed(meters < 100000 ? 3 : 1)} km`;
}

export function formatArea(squareMeters) {
    if (squareMeters < 100000) return `${squareMeters.toFixed(squareMeters < 100 ? 2 : 0)} m²`;
    if (squareMeters < 10000000) return `${(squareMeters / 10000).toFixed(2)} ha`;
    return `${(squareMeters / 1000000).toFixed(2)} km²`;
}

export function formatCoordinate([lon, lat]) {
    const ns = lat >= 0 ? 'N' : 'S';
    const ew = lon >= 0 ? 'E' : 'W';
    return `${Math.abs(lat).toFixed(5)}° ${ns}, ${Math.abs(lon).toFixed(5)}° ${ew}`;
}

/** A short description of a feature's size: length, area and perimeter, or position. */
export function describeGeometry(geometry) {
    if (!geometry) return '';
    if (geometry.type === 'Point') return formatCoordinate(geometry.coordinates);
    if (geometry.type === 'LineString') return `Length: ${formatLength(lineLength(geometry.coordinates))}`;
    if (geometry.type === 'Polygon') {
        return `Area: ${formatArea(polygonArea(geometry.coordinates))}<br>`
            + `Perimeter: ${formatLength(lineLength(geometry.coordinates[0]))}`;
    }
    return '';
}
