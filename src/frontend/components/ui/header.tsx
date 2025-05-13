// creates + styles header for website
//this header is designed to be a white box with a small green border line on the bottom
//the leftside of the header has a green recycling icon
//the center of the header says "Recycle Sort" in green text

import React from 'react'
import { FaRecycle } from 'react-icons/fa'
import * as Toolbar from '@radix-ui/react-toolbar'

const Header = () => {
  return (
    <Toolbar.Root className='w-full border-b-3 border-green-700 p-2 flex items-center relative'>
      <div className='absolute left-6 flex items-center space-x-2'>
        <FaRecycle className='text-green-600 text-4xl' />
      </div>

      <div className='flex-grow text-center'>
        <h1 className='text-4xl text-green-600'>Recycle Sort</h1>
      </div>
    </Toolbar.Root>
  )
}

export default Header
