package com.example.accountapp.controller;

import java.util.List;

import javax.websocket.server.PathParam;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.jpa.repository.Query;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.example.accountapp.exception.AccountNotFoundException;
import com.example.accountapp.model.Account;
import com.example.accountapp.repository.AccountRepository;
import com.example.accountapp.service.AccountService;



@RestController
@RequestMapping("api/account")
public class AccountController {
	
	private AccountService accountService;

	public AccountController(AccountService accountService) {
		super();
		this.accountService = accountService;
	}
	

	
	//create account
	@PostMapping
	public ResponseEntity<Account> saveAccount(@RequestBody Account account){
		return new ResponseEntity<Account>(accountService.saveAccount(account), HttpStatus.CREATED);
		
	}
	
	
	//get all account
	@GetMapping
	public List<Account> getAllAccount(){
		return accountService.getAllAccount();

	}
	
//	//get the sorted list of account
	@GetMapping("/sort")
	public List<Account> getAccountByOrder(){
		 return accountService.getAccountByholdingId();
		
	}

	
	
	//get account by id
	@GetMapping("{id}")
	public ResponseEntity<Account> getAccountById(@PathVariable("id") long id){
		return new ResponseEntity<Account>(accountService.getAccountById(id), HttpStatus.OK);
		
	}
	
	//update a existing account
	@PutMapping("{id}")
	public ResponseEntity<Account> updateAccount(@PathVariable("id") long id,@RequestBody Account account){
		return new ResponseEntity<Account>(accountService.updateAccount(account, id), HttpStatus.OK);
	}

	
	// Delete account
	@DeleteMapping("{id}")
	public ResponseEntity<String> deleteAccount(@PathVariable("id") long id){
		
		accountService.deleteAccount(id);
		
		return new ResponseEntity<String>("Account deleted Sucessfully!",HttpStatus.OK);
		
	}
}
