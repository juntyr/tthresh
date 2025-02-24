/*
 * Copyright (c) 2016-2017, Rafael Ballester-Ripoll
 *                          (Visualization and MultiMedia Lab, University of Zurich),
 *                          rballester@ifi.uzh.ch
 *
 * Licensed under the LGPLv3.0 (https://github.com/rballester/tthresh/blob/master/LICENSE)
 */

#ifndef __IO_HPP__
#define __IO_HPP__

#include <istream>
#include <stdio.h>
#include <string.h>
#include <assert.h>

struct reader {
    uint64_t rbytes;
    int8_t rbit;
    std::istream &input;

    reader(std::istream &input): rbytes(0), rbit(-1), input(input) {}
}; // Read state

struct writer {
    uint64_t wbytes;
    int8_t wbit;
    std::ostream &output;
    size_t total_written_bytes; // Used to compute the final output size

    writer(std::ostream &output): wbytes(0), wbit(-1), output(output), total_written_bytes(0) {}
}; // Write state

/*********/
// Writing
/*********/

// Call open_wbit() before write_bits()
// If write_bits() has been called, call close_wbit() before write_stream()

void write_stream(writer &w, unsigned char *buf, size_t bytes_to_write)
{
    w.output.write(reinterpret_cast<char*>(buf), bytes_to_write);
    w.total_written_bytes += bytes_to_write;
}

void open_wbit(writer &w) {
    w.wbytes = 0;
    w.wbit = 63;
}

// Assumption: to_write <= 64
void write_bits(writer &w, uint64_t bits, char to_write) {
    if (to_write <= w.wbit+1) {
        w.wbytes |= bits << (w.wbit+1-to_write);
        w.wbit -= to_write;
    }
    else {
        if (w.wbit > -1)
            w.wbytes |= bits >> (to_write-(w.wbit+1));
        write_stream(w, reinterpret_cast<unsigned char *> (&w.wbytes), sizeof(w.wbytes));
        to_write -= w.wbit+1;
        w.wbytes = 0;
        w.wbytes |= bits << (64-to_write);
        w.wbit = 63-to_write;
    }
}

void close_wbit(writer &w) {
    // Write any reamining bits
    if (w.wbit < 63)
        write_stream(w, reinterpret_cast < unsigned char *> (&w.wbytes), sizeof(w.wbytes));
}

/*********/
// Reading
/*********/

// If read_bits() has been called, call close_rbit() before read_stream()

void read_stream(reader &r, uint8_t *buf, size_t bytes_to_read)
{
    r.input.read(reinterpret_cast<char *>(buf), bytes_to_read);
    size_t howmany = r.input.gcount();
    if (howmany != bytes_to_read) {
        cout << "Error: tried to read " << bytes_to_read << " bytes, got only " << howmany << endl;
        exit(1);
    }
}

void close_rbit(reader &r)
{
    r.rbytes = 0;
    r.rbit = -1;
}

// Assumption: to_read <= BITS
uint64_t read_bits(reader &r, char to_read) {
    uint64_t result = 0;
    if (to_read <= r.rbit+1) {
        result = r.rbytes << (63-r.rbit) >> (64-to_read);
        r.rbit -= to_read;
    }
    else {
        if (r.rbit > -1)
            result = r.rbytes << (64-r.rbit-1) >> (64-to_read);
        read_stream(r, reinterpret_cast<uint8_t *> (&r.rbytes), sizeof(r.rbytes));
        to_read -= r.rbit+1;
        result |= r.rbytes >> (64-to_read);
        r.rbit = 63-to_read;
    }
    return result;
}

#endif // IO_HPP
