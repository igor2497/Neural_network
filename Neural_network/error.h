#ifndef ERROR_H
#define ERROR_H
typedef enum ERR_ENUM {
	ERR_OK = 0,							//!< no error
	ERR_FAIL,							//!< general error
	ERR_FILE_OPEN,						//!< error while opening a file
	ERR_INVALID_ARG						//!< invalid argument
} ERR_E;
#endif // #ifndef ERROR_H
