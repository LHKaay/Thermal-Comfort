import proj4 from 'proj4/dist/proj4'
import {get} from 'ol/proj'
import {register} from 'ol/proj/proj4'

import { transform } from 'ol/proj4'

var epsg4326 = transform([374490.0156774708262807316944, 283461.9648496026517224702106], 'EPSG:5186', 'EPSG:4326')
console.log(epsg4326)